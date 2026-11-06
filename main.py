from dataclasses import dataclass
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class GAConfig:
    # ---------- ไฟล์ input / output (แก้ตรงนี้ที่เดียว) ----------
    csv_path: Path = Path("HR.csv")  # ไฟล์ข้อมูลพนักงาน เช่น Path("D:/HR.csv")
    output_dir: Path = Path("ga_output")  # โฟลเดอร์ที่เก็บผลลัพธ์

    # ---------- พารามิเตอร์ของ GA ----------
    population_size: int = 160  # จำนวนคำตอบในหนึ่ง generation.
    generations: int = 600  # จำนวนรอบวิวัฒนาการ.
    mutation_rate: float = 0.25  # โอกาสที่ลูกหนึ่งคนจะสลับสมาชิกในทีม.
    elite_size: int = 8  # จำนวนคำตอบดีสุดที่ส่งผ่าน generation โดยไม่เปลี่ยน.
    tournament_size: int = 4  # จำนวนผู้เข้าแข่งที่ใช้เลือกพ่อหรือแม่หนึ่งคน.
    seed: int = 42  # ทำให้ผลสุ่มเดิมเกิดซ้ำได้.


# คอลัมน์ขั้นต่ำที่โจทย์ต้องใช้
REQUIRED_COLUMNS = {"gender", "department", "weight", "height", "age"}
TEAM_COLORS = {"Red": "#d62828", "Green": "#2a9d5b"}
BMI_LABELS = ["<18.5", "18.5-22.9", "23.0-24.9", "25+"]


def read_and_prepare(csv_path: Path) -> pd.DataFrame:
    """อ่านข้อมูล ตรวจสอบความถูกต้อง และเพิ่ม BMI กับรหัสแถว."""
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip().str.lower()
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"CSV ไม่มีคอลัมน์ที่จำเป็น: {sorted(missing)}")
    data = data.copy()
    data["gender"] = data["gender"].astype(str).str.strip()
    data["department"] = data["department"].astype(str).str.strip()
    for column in ("weight", "height", "age"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("พบข้อมูลว่างหรือไม่ใช่ตัวเลขในคอลัมน์ที่จำเป็น")
    if (data["height"] <= 0).any() or (data["weight"] <= 0).any():
        raise ValueError("weight และ height ต้องมากกว่า 0")
    data["employee_row"] = np.arange(1, len(data) + 1)
    data["bmi"] = data["weight"] / (data["height"] / 100) ** 2
    data["bmi_group"] = pd.cut(
        data["bmi"],
        bins=[-np.inf, 18.5, 23.0, 25.0, np.inf],
        labels=BMI_LABELS,
        right=False,
    )
    return data


class Problem:
    """เก็บข้อมูลที่แปลงเป็น numpy ไว้ล่วงหน้า เพื่อให้คำนวณ score ทั้ง population ได้เร็ว."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.n = len(data)

        # ค่าตัวเลขที่ใช้เทียบความเท่าเทียม: age, weight, height, bmi (เรียงเป็นคอลัมน์)
        self.values = data[["age", "weight", "height", "bmi"]].to_numpy(dtype=float)
        self.value_totals = self.values.sum(axis=0)
        ranges = self.values.max(axis=0) - self.values.min(axis=0)
        self.value_ranges = np.maximum(ranges, 1.0)  # normalize + กันหารด้วย 0

        # one-hot ของกลุ่ม BMI และแผนก
        bmi_codes = data["bmi_group"].cat.codes.to_numpy()
        self.bmi_onehot = np.eye(len(BMI_LABELS))[bmi_codes]
        dept_codes, dept_names = pd.factorize(data["department"])
        self.dept_onehot = np.eye(len(dept_names))[dept_codes]
        self.dept_totals = self.dept_onehot.sum(axis=0)
        self.n_departments = len(dept_names)

        # เป้าหมายจำนวนคนทีมแดงของแต่ละเพศ + index ของแต่ละเพศ
        self.targets = make_gender_targets(data)
        self.gender_indices = {
            gender: np.flatnonzero((data["gender"] == gender).to_numpy())
            for gender in self.targets
        }


def make_gender_targets(data: pd.DataFrame) -> dict[str, int]:
    """กำหนดจำนวนของแต่ละเพศที่ต้องอยู่ทีมแดง; ทีมเขียวได้ส่วนที่เหลือ."""
    counts = data["gender"].value_counts()
    if any(count % 2 for count in counts):
        raise ValueError("จำนวนพนักงานของอย่างน้อยหนึ่งเพศเป็นเลขคี่ จึงแบ่งเพศเท่ากันเป๊ะไม่ได้")
    if len(data) % 2 != 0:
        raise ValueError("จำนวนพนักงานรวมเป็นเลขคี่ จึงแบ่งสองสีเท่ากันไม่ได้")
    return {gender: count // 2 for gender, count in counts.items()}


def random_individual(problem: Problem, rng: random.Random) -> np.ndarray:
    """สร้าง chromosome เริ่มต้นที่จำนวนคนและเพศสมดุลตาม targets ตั้งแต่แรก."""
    chromosome = np.zeros(problem.n, dtype=bool)
    for gender, red_count in problem.targets.items():
        indices = problem.gender_indices[gender].tolist()
        chromosome[rng.sample(indices, red_count)] = True
    return chromosome


def score_population(problem: Problem, population: np.ndarray) -> np.ndarray:
    """ประเมิน Penalty ของทุกคำตอบพร้อมกัน (ค่ายิ่งต่ำยิ่งดี).

    population: array รูปร่าง (จำนวนคำตอบ, จำนวนพนักงาน) ชนิด bool; True = ทีมแดง
    """
    red = population.astype(float)
    red_n = red.sum(axis=1, keepdims=True)
    green_n = problem.n - red_n

    # 1. ความต่างของค่าเฉลี่ย อายุ น้ำหนัก ส่วนสูง BMI (normalize ด้วยช่วงค่า)
    red_mean = (red @ problem.values) / red_n
    green_mean = (problem.value_totals - red @ problem.values) / green_n
    gaps = np.abs(red_mean - green_mean) / problem.value_ranges  # (P, 4)
    age_gap, weight_gap, height_gap, bmi_gap = gaps.T

    # 2. สัดส่วนกลุ่ม BMI ของสองทีมต้องใกล้กัน
    red_mix = (red @ problem.bmi_onehot) / red_n
    green_mix = (problem.bmi_onehot.sum(axis=0) - red @ problem.bmi_onehot) / green_n
    bmi_mix_gap = np.abs(red_mix - green_mix).sum(axis=1)

    # 3. เงื่อนไขบังคับ: แผนกที่มี 2 คนขึ้นไปห้ามอยู่สีเดียวทั้งหมด
    red_dept = red @ problem.dept_onehot  # จำนวนคนทีมแดงในแต่ละแผนก
    all_one_side = (red_dept == 0) | (red_dept == problem.dept_totals)
    split_violations = (all_one_side & (problem.dept_totals >= 2)).sum(axis=1)

    # 4. แต่ละทีมควรมีแผนกหลากหลาย
    red_diversity = (red_dept > 0).sum(axis=1)
    green_diversity = ((problem.dept_totals - red_dept) > 0).sum(axis=1)
    min_diversity = max(problem.n_departments - 1, 1)
    diversity_gap = (
        np.maximum(0, min_diversity - red_diversity)
        + np.maximum(0, min_diversity - green_diversity)
    )

    # split_violations ใส่ Penalty 10,000 (Hard Constraint) เพื่อไม่ให้ GA เลือกคำตอบที่ผิดกติกา
    return (
        3.0 * age_gap
        + 3.0 * weight_gap
        + 3.0 * height_gap
        + 3.0 * bmi_gap
        + 2.0 * bmi_mix_gap
        + 10000.0 * split_violations
        + 10.0 * diversity_gap
    )


def count_split_violations(problem: Problem, chromosome: np.ndarray) -> int:
    """นับจำนวนแผนก (ที่มี >= 2 คน) ที่ไปกระจุกอยู่สีเดียวทั้งหมด."""
    red_dept = chromosome.astype(float) @ problem.dept_onehot
    all_one_side = (red_dept == 0) | (red_dept == problem.dept_totals)
    return int((all_one_side & (problem.dept_totals >= 2)).sum())


def tournament_select(population: np.ndarray, scores: np.ndarray, config: GAConfig, rng: random.Random) -> np.ndarray:
    """เลือกพ่อ/แม่ด้วย tournament selection: คะแนน penalty ต่ำกว่าชนะ."""
    candidates = rng.sample(range(len(population)), config.tournament_size)
    winner = min(candidates, key=lambda index: scores[index])
    return population[winner]


def crossover(parent_a: np.ndarray, parent_b: np.ndarray, problem: Problem, rng: random.Random) -> np.ndarray:
    """ผสมพ่อแม่ทีละเพศ แล้วซ่อมจำนวนสมาชิกแดงให้ตรง target เสมอ."""
    child = np.zeros(problem.n, dtype=bool)
    for gender, red_target in problem.targets.items():
        indices = problem.gender_indices[gender].tolist()
        both_red = [i for i in indices if parent_a[i] and parent_b[i]]
        one_red = [i for i in indices if parent_a[i] != parent_b[i]]
        neither = [i for i in indices if not parent_a[i] and not parent_b[i]]
        for pool in (both_red, one_red, neither):
            rng.shuffle(pool)
        # ลำดับความสำคัญ: แดงทั้งคู่ -> แดงฝั่งเดียว -> ไม่แดงเลย
        child[(both_red + one_red + neither)[:red_target]] = True
    return child


def mutate(chromosome: np.ndarray, problem: Problem, rng: random.Random) -> np.ndarray:
    """สลับ Red/Green ของคนเพศเดียวกันหนึ่งคู่ เพื่อรักษาจำนวนเพศให้เท่าเดิม."""
    child = chromosome.copy()
    gender = rng.choice(list(problem.targets))
    gender_indices = problem.gender_indices[gender].tolist()
    red_candidates = [i for i in gender_indices if child[i]]
    green_candidates = [i for i in gender_indices if not child[i]]
    child[rng.choice(red_candidates)] = False
    child[rng.choice(green_candidates)] = True
    return child


def run_ga(problem: Problem, config: GAConfig) -> tuple[np.ndarray, list[float]]:
    """ทำวงจร GA และคืน chromosome ที่ penalty ต่ำสุด พร้อมประวัติ best score."""
    rng = random.Random(config.seed)
    population = np.array([random_individual(problem, rng) for _ in range(config.population_size)])
    history = []
    for _ in range(config.generations):
        scores = score_population(problem, population)
        order = np.argsort(scores)
        history.append(float(scores[order[0]]))
        next_population = [population[index].copy() for index in order[: config.elite_size]]
        while len(next_population) < config.population_size:
            parent_a = tournament_select(population, scores, config, rng)
            parent_b = tournament_select(population, scores, config, rng)
            child = crossover(parent_a, parent_b, problem, rng)
            if rng.random() < config.mutation_rate:
                child = mutate(child, problem, rng)
            next_population.append(child)
        population = np.array(next_population)
    final_scores = score_population(problem, population)
    best = population[int(np.argmin(final_scores))]
    return best, history


def write_outputs(data: pd.DataFrame, chromosome: np.ndarray, history: list[float], output_dir: Path) -> None:
    """สร้าง CSV และกราฟสรุป fairness ของคำตอบที่ GA เลือก."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result = data.copy()
    result["team"] = np.where(chromosome, "Red", "Green")
    result.to_csv(output_dir / "team_assignment.csv", index=False)

    summary = result.groupby("team").agg(
        people=("team", "size"),
        female=("gender", lambda x: (x == "Female").sum()),
        male=("gender", lambda x: (x == "Male").sum()),
        mean_age=("age", "mean"),
        mean_weight=("weight", "mean"),
        mean_height=("height", "mean"),
        mean_bmi=("bmi", "mean"),
        departments=("department", "nunique"),
    ).round(2)
    summary.to_csv(output_dir / "fairness_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    gender_table = pd.crosstab(result["gender"], result["team"]).reindex(columns=["Red", "Green"], fill_value=0)
    gender_table.plot(kind="bar", ax=axes[0, 0], color=[TEAM_COLORS["Red"], TEAM_COLORS["Green"]])
    axes[0, 0].set_title("Gender balance")
    axes[0, 0].set_ylabel("People")
    axes[0, 0].tick_params(axis="x", rotation=0)

    # ค่าเฉลี่ย 4 ตัวชี้วัดของสองทีม
    metrics = result.groupby("team")[["age", "weight", "height", "bmi"]].mean().T
    metrics = metrics.reindex(columns=["Red", "Green"])
    metrics.index = ["Age", "Weight", "Height", "BMI"]
    metrics.plot(kind="bar", ax=axes[0, 1], color=[TEAM_COLORS["Red"], TEAM_COLORS["Green"]])
    axes[0, 1].set_title("Physical Fairness Averages")
    axes[0, 1].set_ylabel("Mean Value")
    axes[0, 1].tick_params(axis="x", rotation=0)

    department_table = pd.crosstab(result["department"], result["team"]).reindex(columns=["Red", "Green"], fill_value=0)
    department_table.plot(kind="bar", stacked=True, ax=axes[1, 0], color=[TEAM_COLORS["Red"], TEAM_COLORS["Green"]])
    axes[1, 0].set_title("Department distribution")
    axes[1, 0].set_ylabel("People")
    axes[1, 0].set_xticks(range(len(department_table.index)))
    axes[1, 0].set_xticklabels(department_table.index, rotation=45, ha="right")

    axes[1, 1].plot(range(1, len(history) + 1), history, color="#264653")
    axes[1, 1].set_title("GA convergence (lower is better)")
    axes[1, 1].set_xlabel("Generation")
    axes[1, 1].set_ylabel("Best penalty")
    axes[1, 1].set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_dir / "fairness_charts.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """อ่านค่าจาก GAConfig, รัน GA และเขียนผลลัพธ์ทั้งหมด."""
    config = GAConfig()
    data = read_and_prepare(config.csv_path)
    problem = Problem(data)
    chromosome, history = run_ga(problem, config)

    violations = count_split_violations(problem, chromosome)
    if violations:
        print(f"คำเตือน: ยังมี {violations} แผนกที่อยู่สีเดียวทั้งหมด ลองเพิ่ม generations หรือ population_size")
    write_outputs(data, chromosome, history, config.output_dir)
    print(f"เสร็จแล้ว: {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()