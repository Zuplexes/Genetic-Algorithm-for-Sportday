"""
แบ่งทีมกีฬาสี แดง / ฟ้า ด้วย Genetic Algorithm
------------------------------------------------
เงื่อนไข
 1) Fairness : จำนวนคน, เพศ, ค่าเฉลี่ย+SD ของ น้ำหนัก / ส่วนสูง / อายุ / BMI ใกล้กัน
              + จำนวนคนในแต่ละกลุ่ม BMI (เกณฑ์เอเชีย) ของสองทีมใกล้กัน
 2) ทุกหน่วยงานที่มี >= 2 คน ต้องมีสมาชิกอยู่ทั้งสองสี (และพยายามแบ่ง ~50/50)

ยีน (chromosome) : เวกเตอร์ 0/1 ยาว N   (0 = แดง, 1 = ฟ้า)
ใช้งาน            : python ga_sports_day.py HR.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

SEED = 42
POP, GENS, ELITE, TOUR_K = 200, 1500, 4, 3
PATIENCE = 300                       # หยุดถ้าไม่ดีขึ้นติดกันกี่รุ่น
W = dict(size=10, gender=10, mean=100, sd=100, dept=3, bmi_cat=3, violate=1000)
BMI_BINS = [0, 18.5, 23, 25, 30, np.inf]                 # เกณฑ์ BMI เอเชีย
BMI_LABELS = ["ผอม", "ปกติ", "ท้วม", "อ้วน", "อ้วนมาก"]

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- data
path = sys.argv[1] if len(sys.argv) > 1 else "HR.csv"
df = pd.read_csv(path, skipinitialspace=True).dropna()
df.columns = df.columns.str.strip()
for c in ("gender", "department"):
    df[c] = df[c].str.strip()
df = df.reset_index(drop=True)
df["bmi"] = (df["weight"] / (df["height"] / 100) ** 2).round(2)
df["bmi_group"] = pd.cut(df["bmi"], BMI_BINS, labels=BMI_LABELS, right=False)

N = len(df)
HALF = N // 2
NUM = ["weight", "height", "age", "bmi"]
Z = ((df[NUM] - df[NUM].mean()) / df[NUM].std()).to_numpy()   # N x 4 (z-score)
male = (df["gender"] == "Male").to_numpy().astype(float)
depts = sorted(df["department"].unique())
D = (df["department"].to_numpy()[:, None] == np.array(depts)[None, :]).astype(float)  # N x nD
dept_n = D.sum(0)
dept_odd = dept_n % 2                    # ส่วนต่างขั้นต่ำที่เลี่ยงไม่ได้ (หน่วยงานจำนวนคี่)
must_split = dept_n >= 2                 # หน่วย B มี 1 คน -> ยกเว้น
B = (df["bmi_group"].astype(str).to_numpy()[:, None] == np.array(BMI_LABELS)[None, :]).astype(float)
bmi_n = B.sum(0)
bmi_odd = bmi_n % 2
strata = df.groupby(["department", "gender"]).indices                       # ใช้ใน mutation
strata_init = df.groupby(["department", "gender", "bmi_group"], observed=True).indices  # ใช้ตอนเริ่ม


# ---------------------------------------------------------------- fitness
def cost(pop):
    """pop: P x N (0/1) -> cost P (ยิ่งต่ำยิ่งดี)"""
    pop = pop.astype(float)
    nb = pop.sum(1)
    nr = N - nb
    size = np.abs(nb - nr)
    mb = pop @ male
    gender = np.abs(mb - (male.sum() - mb))

    # mean & SD ของ z-score ในแต่ละทีม
    sb, sr = pop @ Z, (1 - pop) @ Z
    meanb, meanr = sb / nb[:, None], sr / nr[:, None]
    qb, qr = pop @ (Z ** 2), (1 - pop) @ (Z ** 2)
    sdb = np.sqrt(np.maximum(qb / nb[:, None] - meanb ** 2, 0))
    sdr = np.sqrt(np.maximum(qr / nr[:, None] - meanr ** 2, 0))
    mean_gap = np.abs(meanb - meanr).sum(1)
    sd_gap = np.abs(sdb - sdr).sum(1)

    db = pop @ D
    dr = dept_n - db
    dept_gap = (np.abs(db - dr) - dept_odd).sum(1)
    violate = (((db == 0) | (dr == 0)) & must_split).sum(1)

    bb = pop @ B                                         # จำนวนฟ้าในแต่ละกลุ่ม BMI
    bmi_gap = (np.abs(bb - (bmi_n - bb)) - bmi_odd).sum(1)

    return (W["size"] * size + W["gender"] * gender + W["mean"] * mean_gap
            + W["sd"] * sd_gap + W["dept"] * dept_gap + W["bmi_cat"] * bmi_gap
            + W["violate"] * violate)


# ---------------------------------------------------------------- operators
def stratified_individual():
    """แบ่งครึ่งในทุกกลุ่ม หน่วยงาน x เพศ x กลุ่ม BMI -> เริ่มต้นใกล้สมดุลทุกด้าน"""
    x = np.zeros(N, dtype=np.int8)
    for idx in strata_init.values():
        idx = rng.permutation(idx)
        k = len(idx) // 2 + (rng.random() < 0.5) * (len(idx) % 2)
        x[idx[:k]] = 1
    return repair(x)


def repair(x):
    """ปรับให้ฟ้า = HALF คน (สลับคนจากทีมที่ใหญ่กว่าแบบสุ่ม)"""
    diff = int(x.sum()) - HALF
    if diff > 0:
        x[rng.choice(np.flatnonzero(x == 1), diff, replace=False)] = 0
    elif diff < 0:
        x[rng.choice(np.flatnonzero(x == 0), -diff, replace=False)] = 1
    return x


def tournament(pop, fit):
    idx = rng.integers(0, len(pop), TOUR_K)
    return pop[idx[np.argmin(fit[idx])]]


def crossover(a, b):
    mask = rng.random(N) < 0.5          # uniform crossover
    return repair(np.where(mask, a, b).astype(np.int8))


def mutate(x):
    """swap คนแดง 1 คน <-> ฟ้า 1 คน ในกลุ่ม หน่วยงาน x เพศ เดียวกัน (ส่วนใหญ่)
       หรือ เพศเดียวกันข้ามหน่วยงาน (บางครั้ง) -> จำนวนคนและเพศยังสมดุล"""
    for _ in range(rng.integers(1, 4)):
        if rng.random() < 0.8:
            keys = list(strata.keys())
            idx = strata[keys[rng.integers(len(keys))]]
        else:
            idx = np.flatnonzero(male == rng.integers(0, 2))
        reds, blues = idx[x[idx] == 0], idx[x[idx] == 1]
        if len(reds) and len(blues):
            i, j = rng.choice(reds), rng.choice(blues)
            x[i], x[j] = 1, 0
    return x


# ---------------------------------------------------------------- GA loop
def run_ga():
    pop = np.array([stratified_individual() for _ in range(POP)])
    fit = cost(pop)
    history, best_f, stall = [], np.inf, 0
    for g in range(GENS):
        order = np.argsort(fit)
        new = [pop[i].copy() for i in order[:ELITE]]           # elitism
        while len(new) < POP:
            child = crossover(tournament(pop, fit), tournament(pop, fit))
            new.append(mutate(child))
        pop = np.array(new)
        fit = cost(pop)
        history.append((g + 1, fit.min(), fit.mean()))
        if fit.min() < best_f - 1e-9:
            best_f, stall = fit.min(), 0
        else:
            stall += 1
        if stall >= PATIENCE:
            break
    hist = pd.DataFrame(history, columns=["generation", "best", "mean"]).assign(phase="GA")
    best = pop[np.argmin(fit)].copy()
    best, ls_hist = local_search(best, start_gen=len(hist))
    return best, pd.concat([hist, ls_hist], ignore_index=True)


def local_search(x, start_gen, batch=400, max_fail=40, max_iter=3000):
    """ขัดเกลาคำตอบที่ดีที่สุดของ GA (Hybrid / Memetic GA):
       ลองสลับคนเพศเดียวกันระหว่างแดง<->ฟ้า ทีละ batch แล้วรับเฉพาะการสลับที่ทำให้ cost ลดลง"""
    cur = cost(x[None])[0]
    rows, fail, it = [], 0, 0
    while fail < max_fail and it < max_iter:
        it += 1
        sex = rng.integers(0, 2, batch)
        cand = np.repeat(x[None], batch, axis=0)
        for k in range(batch):
            idx = np.flatnonzero(male == sex[k])
            i = rng.choice(idx[x[idx] == 0]); j = rng.choice(idx[x[idx] == 1])
            cand[k, i], cand[k, j] = 1, 0
        f = cost(cand)
        if f.min() < cur - 1e-9:
            x, cur, fail = cand[np.argmin(f)].copy(), f.min(), 0
        else:
            fail += 1
        rows.append((start_gen + it, cur, np.nan, "Local search"))
    return x, pd.DataFrame(rows, columns=["generation", "best", "mean", "phase"])


def random_split():
    x = np.zeros(N, dtype=np.int8)
    x[rng.choice(N, HALF, replace=False)] = 1
    return x


# ---------------------------------------------------------------- evaluation
def balance_table(x):
    t = df.assign(team=np.where(x == 1, "Blue", "Red"))
    r, b = t[t.team == "Red"], t[t.team == "Blue"]
    rows = [("จำนวนคน", len(r), len(b), np.nan, np.nan)]
    pr, pb = (r.gender == "Male").mean(), (b.gender == "Male").mean()
    p = (pr + pb) / 2
    chi_p = stats.chi2_contingency(pd.crosstab(t.team, t.gender))[1]
    rows.append(("สัดส่วนชาย", round(pr, 3), round(pb, 3),
                 (pb - pr) / np.sqrt(p * (1 - p)), chi_p))
    for c in NUM:
        smd = (b[c].mean() - r[c].mean()) / np.sqrt((r[c].var() + b[c].var()) / 2)
        rows.append((f"{c} (mean±SD)",
                     f"{r[c].mean():.2f} ± {r[c].std():.2f}",
                     f"{b[c].mean():.2f} ± {b[c].std():.2f}",
                     smd, stats.ttest_ind(r[c], b[c]).pvalue))
    ct = pd.crosstab(t.team, t.bmi_group).reindex(["Red", "Blue"])
    rows.append(("กลุ่ม BMI (" + "/".join(BMI_LABELS) + ")",
                 "/".join(map(str, ct.loc["Red"])), "/".join(map(str, ct.loc["Blue"])),
                 np.nan, stats.chi2_contingency(ct)[1]))
    out = pd.DataFrame(rows, columns=["ตัวแปร", "แดง", "ฟ้า", "SMD", "p-value"])
    out["SMD"] = out["SMD"].round(4)
    out["p-value"] = out["p-value"].round(3)
    return out, t


def dept_check(t):
    ct = pd.crosstab(t.department, t.team).reindex(columns=["Red", "Blue"], fill_value=0)
    ct["รวม"] = ct.sum(axis=1)
    ct["ผ่านเงื่อนไข 2"] = np.where(ct["รวม"] < 2, "ยกเว้น (1 คน)",
                                    np.where((ct.Red > 0) & (ct.Blue > 0), "ผ่าน", "ไม่ผ่าน"))
    return ct


if __name__ == "__main__":
    best, hist = run_ga()
    rnd = random_split()
    tab_ga, teams = balance_table(best)
    tab_rnd, _ = balance_table(rnd)
    dep = dept_check(teams)

    n_ga = (hist.phase == "GA").sum()
    print(f"GA {n_ga} generations (cost {hist.best.iloc[n_ga-1]:.3f}) -> local search {len(hist)-n_ga} รอบ (cost {hist.best.iloc[-1]:.3f})")
    print("\n=== GA ===\n", tab_ga.to_string(index=False))
    print("\n=== สุ่มธรรมดา ===\n", tab_rnd.to_string(index=False))
    print("\n=== หน่วยงาน ===\n", dep.to_string())
    print("\n=== กลุ่ม BMI (GA) ===\n",
          pd.crosstab(teams.bmi_group, teams.team).reindex(columns=["Red", "Blue"]).to_string())

    out = teams.assign(สี=teams.team.map({"Red": "แดง", "Blue": "ฟ้า"}))
    out.to_csv("HR_teams.csv", index=False, encoding="utf-8-sig")
    hist.to_csv("ga_history.csv", index=False)
    pd.concat([tab_ga.assign(วิธี="GA"), tab_rnd.assign(วิธี="สุ่มธรรมดา")]) \
        .to_csv("balance_summary.csv", index=False, encoding="utf-8-sig")
