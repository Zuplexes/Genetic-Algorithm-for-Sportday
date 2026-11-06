"""
แบ่งทีมกีฬาสี แดง / ฟ้า ด้วย Genetic Algorithm (Hybrid GA + Local search)
=========================================================================
เงื่อนไข
 1) Fairness : จำนวนคน, เพศ, ค่าเฉลี่ย+SD ของ น้ำหนัก / ส่วนสูง / อายุ / BMI ใกล้กัน
              + จำนวนคนในแต่ละกลุ่ม BMI (เกณฑ์เอเชีย) ของสองทีมใกล้กัน
 2) ทุกหน่วยงานที่มี >= 2 คน ต้องมีสมาชิกอยู่ทั้งสองสี (และพยายามแบ่ง ~50/50)

ยีน (chromosome) : เวกเตอร์ 0/1 ยาว N ช่อง ช่องละ 1 คน   (0 = แดง, 1 = ฟ้า)
ใช้งาน            : python ga_sports_day.py HR.csv
ผลลัพธ์           : HR_teams.csv, ga_history.csv, balance_summary.csv
                    (จากนั้นรัน plot_results.py เพื่อวาดกราฟ)
"""

# =============================================================================
# ส่วนที่ 1 : import ไลบรารี
# =============================================================================
import sys
import numpy as np        # คำนวณแบบเมทริกซ์ -> ให้คะแนนประชากรทั้งรุ่นได้พร้อมกัน (เร็ว)
import pandas as pd       # อ่าน / จัดการตาราง CSV
from scipy import stats   # t-test, chi-square ตอนวัดผล


# =============================================================================
# ส่วนที่ 2 : ค่าตั้งต้น (ปรับจูนได้ตรงนี้)
# =============================================================================
SEED = 42                 # ล็อกการสุ่ม -> รันกี่ครั้งก็ได้ผลเหมือนเดิม
POP = 200                 # จำนวนคำตอบ (วิธีแบ่งทีม) ในประชากร 1 รุ่น
GENS = 1500               # จำนวนรุ่นสูงสุด
ELITE = 4                 # จำนวนตัวที่ดีที่สุดที่ส่งต่อไปรุ่นถัดไปโดยไม่แตะ (elitism)
TOUR_K = 3                # tournament selection: สุ่มมาแข่งกันครั้งละ 3 ตัว
PATIENCE = 300            # ถ้า cost ดีที่สุดไม่ลดลงติดกันกี่รุ่น -> หยุด GA

# น้ำหนักความสำคัญของแต่ละเกณฑ์ใน fitness (ยิ่งมากยิ่งสำคัญ)
W = dict(
    size=10,       # จำนวนคนสองทีมต่างกัน
    gender=10,     # จำนวนผู้ชายสองทีมต่างกัน
    mean=100,      # ค่าเฉลี่ย (z-score) ต่างกัน  -> ค่าดิบเล็ก (~0.01) จึงต้องคูณเยอะ
    sd=100,        # SD (z-score) ต่างกัน          -> ให้การกระจายใกล้กัน ไม่ใช่แค่ค่าเฉลี่ย
    dept=3,        # แต่ละหน่วยงานแบ่งห่างจาก 50/50
    bmi_cat=3,     # จำนวนคนแต่ละกลุ่ม BMI ต่างกัน
    violate=1000,  # โทษหนัก: มีหน่วยงานอยู่สีเดียวทั้งหมด (ผิดเงื่อนไข 2)
)

BMI_BINS = [0, 18.5, 23, 25, 30, np.inf]                 # จุดตัด BMI เกณฑ์เอเชีย
BMI_LABELS = ["ผอม", "ปกติ", "ท้วม", "อ้วน", "อ้วนมาก"]

rng = np.random.default_rng(SEED)   # ตัวสุ่มตัวเดียว ใช้ทั้งโปรแกรม


# =============================================================================
# ส่วนที่ 3 : เตรียมข้อมูล
# =============================================================================
# --- อ่านไฟล์และทำความสะอาด (CSV มีช่องว่างหลังจุลภาค เช่น "Male, R")
path = sys.argv[1] if len(sys.argv) > 1 else "HR.csv"
df = pd.read_csv(path, skipinitialspace=True).dropna()
df.columns = df.columns.str.strip()
for c in ("gender", "department"):
    df[c] = df[c].str.strip()
df = df.reset_index(drop=True)

# --- คำนวณ BMI = น้ำหนัก(kg) / ส่วนสูง(m)^2  แล้วจัดกลุ่มตามเกณฑ์เอเชีย
df["bmi"] = (df["weight"] / (df["height"] / 100) ** 2).round(2)
df["bmi_group"] = pd.cut(df["bmi"], BMI_BINS, labels=BMI_LABELS, right=False)

N = len(df)          # จำนวนคนทั้งหมด (358)
HALF = N // 2        # เป้าหมายจำนวนคนทีมฟ้า (179)

# --- แปลงตัวเลขเป็น z-score = (ค่า - เฉลี่ย) / SD
#     ทำให้ น้ำหนัก / ส่วนสูง / อายุ / BMI อยู่สเกลเดียวกัน (1 หน่วย = 1 SD) เทียบกันได้ยุติธรรม
NUM = ["weight", "height", "age", "bmi"]
Z = ((df[NUM] - df[NUM].mean()) / df[NUM].std()).to_numpy()   # เมทริกซ์ N x 4

# --- เพศ: เวกเตอร์ 1/0 (1 = ชาย)
male = (df["gender"] == "Male").to_numpy().astype(float)

# --- หน่วยงาน: เมทริกซ์ one-hot  N x 19  (แถว = คน, คอลัมน์ = หน่วยงาน, 1 = อยู่หน่วยนั้น)
depts = sorted(df["department"].unique())
D = (df["department"].to_numpy()[:, None] == np.array(depts)[None, :]).astype(float)
dept_n = D.sum(0)                 # จำนวนคนในแต่ละหน่วยงาน
dept_odd = dept_n % 2             # หน่วยงานจำนวนคี่ แบ่งเท่ากันไม่ได้ ต่างกันอย่างน้อย 1 -> ไม่ลงโทษส่วนนี้
must_split = dept_n >= 2          # หน่วยงานที่ต้องมีทั้งสองสี (หน่วย B มี 1 คน -> ยกเว้น)

# --- กลุ่ม BMI: เมทริกซ์ one-hot  N x 5  (แนวคิดเดียวกับหน่วยงาน)
B = (df["bmi_group"].astype(str).to_numpy()[:, None] == np.array(BMI_LABELS)[None, :]).astype(float)
bmi_n = B.sum(0)                  # จำนวนคนในแต่ละกลุ่ม BMI
bmi_odd = bmi_n % 2               # กลุ่มจำนวนคี่ -> ส่วนต่างขั้นต่ำ 1

# --- กลุ่มย่อย (strata) : dict {กลุ่ม: array ของ index คนในกลุ่มนั้น}
strata = df.groupby(["department", "gender"]).indices                                    # ใช้ใน mutation
strata_init = df.groupby(["department", "gender", "bmi_group"], observed=True).indices   # ใช้ตอนสร้างประชากรเริ่มต้น


# =============================================================================
# ส่วนที่ 4 : Fitness function  (cost ยิ่งต่ำยิ่งดี)
# =============================================================================
def cost(pop):
    """ให้คะแนนคำตอบ 'ทั้งประชากร' พร้อมกันด้วยการคูณเมทริกซ์
       pop : เมทริกซ์ P x N (0/1)  ->  คืนค่า cost ยาว P

       ตัวอย่าง  pop @ male : ยีน [1,0,1,1] กับ male [1,1,0,1] -> 1+0+0+1 = 2
                 แปลว่า ทีมฟ้ามีผู้ชาย 2 คน
    """
    pop = pop.astype(float)

    # (1) จำนวนคนสองทีมต่างกันเท่าไร
    nb = pop.sum(1)                       # จำนวนคนทีมฟ้า
    nr = N - nb                           # จำนวนคนทีมแดง
    size = np.abs(nb - nr)

    # (2) จำนวนผู้ชายสองทีมต่างกันเท่าไร
    mb = pop @ male                       # ผู้ชายในทีมฟ้า
    gender = np.abs(mb - (male.sum() - mb))

    # (3) ค่าเฉลี่ยและ SD ของ z-score ในแต่ละทีม (ทั้ง 4 ตัวแปรพร้อมกัน)
    sb, sr = pop @ Z, (1 - pop) @ Z                       # ผลรวมของแต่ละทีม
    meanb, meanr = sb / nb[:, None], sr / nr[:, None]     # ค่าเฉลี่ย
    qb, qr = pop @ (Z ** 2), (1 - pop) @ (Z ** 2)         # ผลรวมกำลังสอง
    # SD จากสูตร Var = E[x^2] - (E[x])^2   (np.maximum กันค่าติดลบเล็กๆ จากการปัดเศษ)
    sdb = np.sqrt(np.maximum(qb / nb[:, None] - meanb ** 2, 0))
    sdr = np.sqrt(np.maximum(qr / nr[:, None] - meanr ** 2, 0))
    mean_gap = np.abs(meanb - meanr).sum(1)               # รวมส่วนต่างค่าเฉลี่ยทุกตัวแปร
    sd_gap = np.abs(sdb - sdr).sum(1)                     # รวมส่วนต่าง SD ทุกตัวแปร

    # (4) หน่วยงาน: แบ่งห่างจาก 50/50 เท่าไร (หักส่วนที่เลี่ยงไม่ได้ของหน่วยเลขคี่)
    db = pop @ D                          # คนฟ้าในแต่ละหน่วยงาน
    dr = dept_n - db                      # คนแดงในแต่ละหน่วยงาน
    dept_gap = (np.abs(db - dr) - dept_odd).sum(1)

    # (5) เงื่อนไข 2: นับหน่วยงานที่อยู่สีเดียวทั้งหมด
    violate = (((db == 0) | (dr == 0)) & must_split).sum(1)

    # (6) กลุ่ม BMI: จำนวนคนแต่ละกลุ่มสองทีมต่างกันเท่าไร
    bb = pop @ B                          # คนฟ้าในแต่ละกลุ่ม BMI
    bmi_gap = (np.abs(bb - (bmi_n - bb)) - bmi_odd).sum(1)

    # รวมทุกเกณฑ์ x น้ำหนักความสำคัญ
    return (W["size"] * size + W["gender"] * gender + W["mean"] * mean_gap
            + W["sd"] * sd_gap + W["dept"] * dept_gap + W["bmi_cat"] * bmi_gap
            + W["violate"] * violate)


# =============================================================================
# ส่วนที่ 5 : ตัวดำเนินการของ GA (operators)
# =============================================================================
def stratified_individual():
    """สร้างคำตอบเริ่มต้น 1 แบบ
       แบ่งครึ่งในทุกกลุ่ม หน่วยงาน x เพศ x กลุ่ม BMI -> เริ่มต้นใกล้สมดุลเกือบทุกด้าน
       (GA จะได้ไม่ต้องเสียเวลาแก้เรื่องพื้นฐาน เหลือจูนค่าตัวเลขอย่างเดียว)"""
    x = np.zeros(N, dtype=np.int8)
    for idx in strata_init.values():
        idx = rng.permutation(idx)                    # สลับลำดับคนในกลุ่มแบบสุ่ม
        # ครึ่งหนึ่งไปฟ้า; ถ้ากลุ่มเป็นเลขคี่ คนที่เหลือ 1 คนสุ่มว่าไปแดงหรือฟ้า
        k = len(idx) // 2 + (rng.random() < 0.5) * (len(idx) % 2)
        x[idx[:k]] = 1
    return repair(x)


def repair(x):
    """ซ่อมให้ทีมฟ้ามี HALF คนพอดี (179/179)
       เกินก็สุ่มย้ายไปแดง ขาดก็สุ่มย้ายมาจากแดง (ใช้หลัง crossover ที่ทำให้จำนวนเพี้ยน)"""
    diff = int(x.sum()) - HALF
    if diff > 0:
        x[rng.choice(np.flatnonzero(x == 1), diff, replace=False)] = 0
    elif diff < 0:
        x[rng.choice(np.flatnonzero(x == 0), -diff, replace=False)] = 1
    return x


def tournament(pop, fit):
    """Tournament selection: สุ่มหยิบ TOUR_K ตัว แล้วคืนตัวที่ cost ต่ำสุด
       ตัวดีมีโอกาสถูกเลือกมากกว่า แต่ตัวรองๆ ก็ยังมีโอกาส -> ประชากรไม่ซ้ำกันเร็วเกินไป"""
    idx = rng.integers(0, len(pop), TOUR_K)
    return pop[idx[np.argmin(fit[idx])]]


def crossover(a, b):
    """Uniform crossover: แต่ละช่องสุ่มเอายีนจากพ่อ (a) หรือแม่ (b) อย่างละครึ่ง
       แล้ว repair ให้จำนวนคนกลับมา 179/179"""
    mask = rng.random(N) < 0.5
    return repair(np.where(mask, a, b).astype(np.int8))


def mutate(x):
    """Mutation แบบ swap: สลับคนแดง 1 คน <-> ฟ้า 1 คน  (ทำ 1-3 คู่)
       - 80% : สลับในกลุ่ม หน่วยงาน x เพศ เดียวกัน -> จำนวนคน/เพศ/หน่วยงานไม่เสีย
               แต่ น้ำหนัก/ส่วนสูง/อายุ/BMI เปลี่ยนได้
       - 20% : สลับเพศเดียวกันข้ามหน่วยงาน -> ช่วยแก้กรณีหน่วยงานแบ่งไม่เท่ากัน"""
    for _ in range(rng.integers(1, 4)):
        if rng.random() < 0.8:
            keys = list(strata.keys())
            idx = strata[keys[rng.integers(len(keys))]]          # สุ่มเลือก 1 กลุ่ม หน่วยงาน x เพศ
        else:
            idx = np.flatnonzero(male == rng.integers(0, 2))     # สุ่มเลือกเพศ (ทุกหน่วยงาน)
        reds, blues = idx[x[idx] == 0], idx[x[idx] == 1]
        if len(reds) and len(blues):                             # ต้องมีทั้งแดงและฟ้าในกลุ่มถึงจะสลับได้
            i, j = rng.choice(reds), rng.choice(blues)
            x[i], x[j] = 1, 0
    return x


# =============================================================================
# ส่วนที่ 6 : วงจรหลักของ GA
# =============================================================================
def run_ga():
    """ขั้นตอน
       สร้างประชากร POP ตัว -> ให้คะแนน
       วนทุกรุ่น:
         (1) เก็บ ELITE ตัวที่ดีที่สุดไว้
         (2) ที่เหลือ: tournament เลือกพ่อแม่ -> crossover -> mutate
         (3) ให้คะแนนรุ่นใหม่ + บันทึก history
         (4) ไม่ดีขึ้นติดกัน PATIENCE รุ่น -> หยุด
       แล้วส่งตัวที่ดีที่สุดไปขัดเกลาต่อด้วย local_search"""
    pop = np.array([stratified_individual() for _ in range(POP)])
    fit = cost(pop)
    history, best_f, stall = [], np.inf, 0

    for g in range(GENS):
        order = np.argsort(fit)                                  # เรียงจากดีที่สุด (cost ต่ำ) -> แย่สุด
        new = [pop[i].copy() for i in order[:ELITE]]             # (1) elitism
        while len(new) < POP:                                    # (2) สร้างลูกจนครบ POP
            child = crossover(tournament(pop, fit), tournament(pop, fit))
            new.append(mutate(child))
        pop = np.array(new)
        fit = cost(pop)                                          # (3) ให้คะแนนรุ่นใหม่
        history.append((g + 1, fit.min(), fit.mean()))

        # (4) เช็กว่าดีขึ้นไหม ถ้าไม่ดีขึ้นนานเกินไปให้หยุด
        if fit.min() < best_f - 1e-9:
            best_f, stall = fit.min(), 0
        else:
            stall += 1
        if stall >= PATIENCE:
            break

    hist = pd.DataFrame(history, columns=["generation", "best", "mean"]).assign(phase="GA")
    best = pop[np.argmin(fit)].copy()
    best, ls_hist = local_search(best, start_gen=len(hist))     # ขัดเกลาต่อ (ส่วนที่ 7)
    return best, pd.concat([hist, ls_hist], ignore_index=True)


# =============================================================================
# ส่วนที่ 7 : Local search ขัดเกลาคำตอบ (Hybrid / Memetic GA)
# =============================================================================
def local_search(x, start_gen, batch=400, max_fail=40, max_iter=3000):
    """GA เก่งเรื่องหา 'โซน' คำตอบที่ดี แต่ช่วงท้ายเข้าใกล้จุดที่ดีที่สุดได้ช้า
       ขั้นนี้ช่วยเก็บรายละเอียด:
         - สร้างแบบใหม่ batch แบบ แต่ละแบบ = คำตอบปัจจุบันที่สลับคนเพศเดียวกัน 1 คู่ (แดง <-> ฟ้า)
         - ให้คะแนนทั้งหมดพร้อมกัน
         - ถ้าแบบที่ดีที่สุดดีกว่าเดิม -> ใช้แทน ; ถ้าไม่ -> นับล้มเหลว
         - ล้มเหลวติดกัน max_fail ครั้ง -> หยุด"""
    cur = cost(x[None])[0]
    rows, fail, it = [], 0, 0
    while fail < max_fail and it < max_iter:
        it += 1
        sex = rng.integers(0, 2, batch)                          # สุ่มเพศของคู่ที่จะสลับ
        cand = np.repeat(x[None], batch, axis=0)                 # สำเนาคำตอบปัจจุบัน batch ชุด
        for k in range(batch):
            idx = np.flatnonzero(male == sex[k])
            i = rng.choice(idx[x[idx] == 0]); j = rng.choice(idx[x[idx] == 1])
            cand[k, i], cand[k, j] = 1, 0                        # สลับ แดง i -> ฟ้า, ฟ้า j -> แดง
        f = cost(cand)
        if f.min() < cur - 1e-9:                                 # ดีขึ้น -> รับ
            x, cur, fail = cand[np.argmin(f)].copy(), f.min(), 0
        else:                                                    # ไม่ดีขึ้น -> นับล้มเหลว
            fail += 1
        rows.append((start_gen + it, cur, np.nan, "Local search"))
    return x, pd.DataFrame(rows, columns=["generation", "best", "mean", "phase"])


# =============================================================================
# ส่วนที่ 8 : วัดผล
# =============================================================================
def random_split():
    """สุ่มแบ่ง 179/179 ธรรมดา (ไม่สนเงื่อนไข) ไว้เป็นตัวเทียบว่า GA ดีกว่าจริง"""
    x = np.zeros(N, dtype=np.int8)
    x[rng.choice(N, HALF, replace=False)] = 1
    return x


def balance_table(x):
    """ตารางเทียบสองทีม
       - SMD = (เฉลี่ยฟ้า - เฉลี่ยแดง) / SD รวม   -> |SMD| < 0.1 ถือว่าสมดุล
       - p-value: t-test (ตัวเลข), chi-square (เพศ, กลุ่ม BMI) -> ค่าสูง = ไม่ต่างกันอย่างมีนัยสำคัญ"""
    t = df.assign(team=np.where(x == 1, "Blue", "Red"))
    r, b = t[t.team == "Red"], t[t.team == "Blue"]
    rows = [("จำนวนคน", len(r), len(b), np.nan, np.nan)]

    # เพศ: SMD ของสัดส่วน + chi-square
    pr, pb = (r.gender == "Male").mean(), (b.gender == "Male").mean()
    p = (pr + pb) / 2
    chi_p = stats.chi2_contingency(pd.crosstab(t.team, t.gender))[1]
    rows.append(("สัดส่วนชาย", round(pr, 3), round(pb, 3),
                 (pb - pr) / np.sqrt(p * (1 - p)), chi_p))

    # ตัวแปรตัวเลข: mean ± SD, SMD, t-test
    for c in NUM:
        smd = (b[c].mean() - r[c].mean()) / np.sqrt((r[c].var() + b[c].var()) / 2)
        rows.append((f"{c} (mean±SD)",
                     f"{r[c].mean():.2f} ± {r[c].std():.2f}",
                     f"{b[c].mean():.2f} ± {b[c].std():.2f}",
                     smd, stats.ttest_ind(r[c], b[c]).pvalue))

    # กลุ่ม BMI: จำนวนคนแต่ละกลุ่ม + chi-square
    ct = pd.crosstab(t.team, t.bmi_group).reindex(["Red", "Blue"])
    rows.append(("กลุ่ม BMI (" + "/".join(BMI_LABELS) + ")",
                 "/".join(map(str, ct.loc["Red"])), "/".join(map(str, ct.loc["Blue"])),
                 np.nan, stats.chi2_contingency(ct)[1]))

    out = pd.DataFrame(rows, columns=["ตัวแปร", "แดง", "ฟ้า", "SMD", "p-value"])
    out["SMD"] = out["SMD"].round(4)
    out["p-value"] = out["p-value"].round(3)
    return out, t


def dept_check(t):
    """ตารางนับคนแดง/ฟ้าในแต่ละหน่วยงาน + สถานะเงื่อนไข 2 (ผ่าน / ไม่ผ่าน / ยกเว้น)"""
    ct = pd.crosstab(t.department, t.team).reindex(columns=["Red", "Blue"], fill_value=0)
    ct["รวม"] = ct.sum(axis=1)
    ct["ผ่านเงื่อนไข 2"] = np.where(ct["รวม"] < 2, "ยกเว้น (1 คน)",
                                    np.where((ct.Red > 0) & (ct.Blue > 0), "ผ่าน", "ไม่ผ่าน"))
    return ct


# =============================================================================
# ส่วนที่ 9 : โปรแกรมหลัก
# =============================================================================
if __name__ == "__main__":
    # --- รัน GA (+ local search) และสุ่มแบ่งธรรมดาไว้เทียบ
    best, hist = run_ga()
    rnd = random_split()
    tab_ga, teams = balance_table(best)
    tab_rnd, _ = balance_table(rnd)
    dep = dept_check(teams)

    # --- พิมพ์ผลออกหน้าจอ
    n_ga = (hist.phase == "GA").sum()
    print(f"GA {n_ga} generations (cost {hist.best.iloc[n_ga-1]:.3f}) -> local search {len(hist)-n_ga} รอบ (cost {hist.best.iloc[-1]:.3f})")
    print("\n=== GA ===\n", tab_ga.to_string(index=False))
    print("\n=== สุ่มธรรมดา ===\n", tab_rnd.to_string(index=False))
    print("\n=== หน่วยงาน ===\n", dep.to_string())
    print("\n=== กลุ่ม BMI (GA) ===\n",
          pd.crosstab(teams.bmi_group, teams.team).reindex(columns=["Red", "Blue"]).to_string())

    # --- บันทึกไฟล์ (utf-8-sig ให้ Excel แสดงภาษาไทยได้)
    out = teams.assign(สี=teams.team.map({"Red": "แดง", "Blue": "ฟ้า"}))
    out.to_csv("HR_teams.csv", index=False, encoding="utf-8-sig")          # รายชื่อ + สี + BMI
    hist.to_csv("ga_history.csv", index=False)                              # cost รายรุ่น (กราฟ 1)
    pd.concat([tab_ga.assign(วิธี="GA"), tab_rnd.assign(วิธี="สุ่มธรรมดา")]) \
        .to_csv("balance_summary.csv", index=False, encoding="utf-8-sig")  # ตารางเทียบ GA vs สุ่ม
