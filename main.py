from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

COLORS = ["#d62828", "#2a9d5b"]  # Red, Green
BMI_LABELS = ["<18.5", "18.5-22.9", "23.0-24.9", "25+"]


def load_data(path):
    """อ่านและตรวจข้อมูล แล้วเพิ่ม bmi กับ bmi_group."""
    d = pd.read_csv(path)
    d.columns = d.columns.str.strip().str.lower()
    cols = ["gender", "department", "weight", "height", "age"]
    if set(cols) - set(d.columns):
        raise ValueError(f"CSV ไม่มีคอลัมน์: {sorted(set(cols) - set(d.columns))}")
    d[["gender", "department"]] = d[["gender", "department"]].astype(str).apply(lambda s: s.str.strip())
    d[["weight", "height", "age"]] = d[["weight", "height", "age"]].apply(pd.to_numeric, errors="coerce")
    if d[cols].isna().any().any() or (d[["weight", "height"]] <= 0).any().any():
        raise ValueError("ข้อมูลว่าง ไม่ใช่ตัวเลข หรือ weight/height <= 0")
    d["bmi"] = d["weight"] / (d["height"] / 100) ** 2
    d["bmi_group"] = pd.cut(d["bmi"], [-np.inf, 18.5, 23, 25, np.inf], labels=BMI_LABELS, right=False)
    return d


def run_ga(d, c):
    """รัน GA คืน (chromosome ที่ดีที่สุด, ประวัติ penalty); True = ทีมแดง."""
    rng, n = random.Random(c["seed"]), len(d)
    counts = d["gender"].value_counts()
    if (counts % 2).any() or n % 2:
        raise ValueError("จำนวนคนรวมหรือของบางเพศเป็นเลขคี่ แบ่งเท่ากันเป๊ะไม่ได้")
    groups = [(np.flatnonzero(d["gender"] == g).tolist(), k // 2) for g, k in counts.items()]

    vals = d[["age", "weight", "height", "bmi"]].to_numpy(float)
    vsum, vrange = vals.sum(0), np.maximum(np.ptp(vals, axis=0), 1.0)
    bmi_oh = np.eye(4)[d["bmi_group"].cat.codes]
    codes, names = pd.factorize(d["department"])
    dept_oh = np.eye(len(names))[codes]
    dept_n = dept_oh.sum(0)

    def score(pop):  # penalty ของทุกคำตอบพร้อมกัน (ยิ่งต่ำยิ่งดี)
        r = pop.astype(float)
        rn, gn = r.sum(1, keepdims=True), n - r.sum(1, keepdims=True)
        gap = np.abs(r @ vals / rn - (vsum - r @ vals) / gn) / vrange
        mix = np.abs(r @ bmi_oh / rn - (bmi_oh.sum(0) - r @ bmi_oh) / gn).sum(1)
        red_dept = r @ dept_oh
        split = (((red_dept == 0) | (red_dept == dept_n)) & (dept_n >= 2)).sum(1)  # แผนกกระจุกสีเดียว
        need = max(len(names) - 1, 1)
        div = np.maximum(0, need - (red_dept > 0).sum(1)) + np.maximum(0, need - ((dept_n - red_dept) > 0).sum(1))
        return 3 * gap.sum(1) + 2 * mix + 10000 * split + 10 * div

    def make():
        x = np.zeros(n, bool)
        for idx, k in groups:
            x[rng.sample(idx, k)] = True
        return x

    def cross(a, b):  # ผสมทีละเพศ ให้จำนวนแดงตรงเป้าเสมอ
        x = np.zeros(n, bool)
        for idx, k in groups:
            both = [i for i in idx if a[i] and b[i]]
            one = [i for i in idx if a[i] != b[i]]
            none = [i for i in idx if not a[i] and not b[i]]
            for p in (both, one, none):
                rng.shuffle(p)
            x[(both + one + none)[:k]] = True
        return x

    def mutate(x):  # สลับแดง/เขียวของคนเพศเดียวกัน 1 คู่
        x, (idx, _) = x.copy(), rng.choice(groups)
        x[rng.choice([i for i in idx if x[i]])] = False
        x[rng.choice([i for i in idx if not x[i]])] = True
        return x

    def pick(pop, s):  # tournament selection
        return pop[min(rng.sample(range(len(pop)), c["tournament_size"]), key=lambda i: s[i])]

    pop, history = np.array([make() for _ in range(c["population_size"])]), []
    for _ in range(c["generations"]):
        s = score(pop)
        order = np.argsort(s)
        history.append(float(s[order[0]]))
        nxt = [pop[i].copy() for i in order[: c["elite_size"]]]
        while len(nxt) < c["population_size"]:
            child = cross(pick(pop, s), pick(pop, s))
            nxt.append(mutate(child) if rng.random() < c["mutation_rate"] else child)
        pop = np.array(nxt)
    return pop[int(np.argmin(score(pop)))], history


def save_outputs(d, best, history, out):
    """เขียน CSV สรุปผล และกราฟ fairness."""
    out.mkdir(parents=True, exist_ok=True)
    d = d.assign(team=np.where(best, "Red", "Green"))
    d.to_csv(out / "team_assignment.csv", index=False)
    g = d.groupby("team")
    summary = g.agg(people=("team", "size"), mean_age=("age", "mean"), mean_weight=("weight", "mean"),
                    mean_height=("height", "mean"), mean_bmi=("bmi", "mean"), departments=("department", "nunique"))
    summary = summary.join(pd.crosstab(d["team"], d["gender"])).round(2)
    summary.to_csv(out / "fairness_summary.csv")

    order = ["Red", "Green"]
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    pd.crosstab(d["gender"], d["team"])[order].plot.bar(ax=ax[0, 0], color=COLORS, rot=0, title="Gender balance", ylabel="People")
    m = g[["age", "weight", "height", "bmi"]].mean().T[order]
    m.index = ["Age", "Weight", "Height", "BMI"]
    m.plot.bar(ax=ax[0, 1], color=COLORS, rot=0, title="Physical Fairness Averages", ylabel="Mean Value")
    pd.crosstab(d["department"], d["team"])[order].plot.bar(ax=ax[1, 0], stacked=True, color=COLORS, rot=45, title="Department distribution", ylabel="People")
    ax[1, 1].plot(range(1, len(history) + 1), history, color="#264653")
    ax[1, 1].set(title="GA convergence (lower is better)", xlabel="Generation", ylabel="Best penalty", yscale="log")

    for c in ax[0, 0].containers:
        ax[0, 0].bar_label(c, fmt="%d", padding=2)
    for c in ax[0, 1].containers:
        ax[0, 1].bar_label(c, fmt="%.2f", padding=2)
    for c in ax[1, 0].containers:  # stacked: ซ่อนเลข 0
        ax[1, 0].bar_label(c, labels=[f"{v:.0f}" if v else "" for v in c.datavalues], label_type="center")
    for x, y in ((1, history[0]), (len(history), history[-1])):  # ค่าเริ่มต้นและค่าสุดท้ายของเส้น
        ax[1, 1].annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")
    fig.tight_layout()
    fig.savefig(out / "fairness_charts.png", dpi=160)
    plt.close(fig)


def main():
    c = yaml.safe_load((Path(__file__).with_name("config.yml")).read_text(encoding="utf-8"))
    d = load_data(c["csv_path"])
    best, history = run_ga(d, c)
    save_outputs(d, best, history, Path(c["output_dir"]))
    print(f"เสร็จแล้ว: {Path(c['output_dir']).resolve()}")


if __name__ == "__main__":
    main()