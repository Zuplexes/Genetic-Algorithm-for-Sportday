"""วาดกราฟสรุปผลจาก ga_sports_day.py  (รันหลัง ga_sports_day.py)"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ---- style (Thai font + calm chrome)
thai = [f.name for f in fm.fontManager.ttflist if f.name in ("Sarabun", "Tahoma", "Leelawadee UI", "Loma", "Garuda")]
plt.rcParams.update({
    "font.family": [thai[0], "DejaVu Sans"] if thai else "sans-serif",
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": "#0b0b0b",
    "axes.titlelocation": "left", "font.size": 10,
})
RED, BLUE, INK, INK2, MUTED = "#e34948", "#2a78d6", "#0b0b0b", "#52514e", "#a3a29b"
TEAMS = ["Red", "Blue"]
TEAM_C = {"Red": RED, "Blue": BLUE}
TEAM_TH = {"Red": "แดง", "Blue": "ฟ้า"}
BMI_LABELS = ["ผอม", "ปกติ", "ท้วม", "อ้วน", "อ้วนมาก"]

t = pd.read_csv("HR_teams.csv")
hist = pd.read_csv("ga_history.csv")
summ = pd.read_csv("balance_summary.csv")


def team_bars(ax, ct, cats, ticklabels):
    """grouped bar แดง/ฟ้า พร้อมตัวเลขบนแท่ง"""
    x = np.arange(len(cats)); w = 0.38
    for i, tm in enumerate(TEAMS):
        bars = ax.bar(x + (i - 0.5) * (w + 0.02), ct.loc[cats, tm], w, color=TEAM_C[tm], label=f"ทีม{TEAM_TH[tm]}")
        ax.bar_label(bars, padding=3, color=INK, fontsize=9)
    ax.set_xticks(x, ticklabels); ax.set_ylabel("จำนวนคน")
    ax.grid(axis="x", visible=False); ax.set_ylim(0, ct.values.max() * 1.15)
    ax.legend(frameon=False)


fig = plt.figure(figsize=(17, 16))
gs = fig.add_gridspec(3, 12, height_ratios=[1, 1, 1.05], hspace=0.45, wspace=1.4, top=0.92)

# 1) fitness curve (GA + local search)
ax = fig.add_subplot(gs[0, 0:4])
ga_h, ls_h = hist[hist.phase == "GA"], hist[hist.phase != "GA"]
ax.plot(ga_h.generation, ga_h["mean"], color=MUTED, lw=1.5, label="cost เฉลี่ยของประชากร (GA)")
ax.plot(ga_h.generation, ga_h.best, color=INK, lw=2, label="cost ดีที่สุด (GA)")
ax.plot(ls_h.generation, ls_h.best, color=INK, lw=2, ls=(0, (3, 1.5)), label="Local search ขัดเกลา")
ax.axvline(ga_h.generation.iloc[-1], color=INK2, lw=0.8, ls=":")
ax.set_yscale("log")
ax.set_xlabel("Generation / รอบ"); ax.set_ylabel("Cost (log scale, ยิ่งต่ำยิ่งดี)")
ax.set_title("1) การลู่เข้าของ GA + Local search")
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.annotate(f"{ga_h.best.iloc[-1]:.1f}", (ga_h.generation.iloc[-1], ga_h.best.iloc[-1]),
            xytext=(-4, 6), textcoords="offset points", ha="right", color=INK2, fontsize=9)
ax.annotate(f"{hist.best.iloc[-1]:.2f}", (hist.generation.iloc[-1], hist.best.iloc[-1]),
            xytext=(5, 0), textcoords="offset points", ha="left", va="center", color=INK, fontsize=9)

# 2) gender per team
ax = fig.add_subplot(gs[0, 4:8])
team_bars(ax, pd.crosstab(t.gender, t.team), ["Male", "Female"], ["ชาย", "หญิง"])
ax.set_title("2) จำนวนชาย / หญิง แต่ละสี")

# 3) SMD GA vs random
ax = fig.add_subplot(gs[0, 8:12])
s = summ[summ["SMD"].notna()]
labels = ["สัดส่วนชาย", "น้ำหนัก", "ส่วนสูง", "อายุ", "BMI"]
y = np.arange(len(labels))
ga = s[s["วิธี"] == "GA"]["SMD"].abs().to_numpy()
rd = s[s["วิธี"] != "GA"]["SMD"].abs().to_numpy()
ax.barh(y + 0.2, rd, 0.36, color=MUTED, label="สุ่มธรรมดา")
ax.barh(y - 0.2, ga, 0.36, color=INK, label="GA")
for yi, v in zip(y + 0.2, rd): ax.text(v + 0.003, yi, f"{v:.3f}", va="center", color=INK2, fontsize=9)
for yi, v in zip(y - 0.2, ga): ax.text(v + 0.003, yi, f"{v:.3f}", va="center", color=INK, fontsize=9)
xmax = max(rd.max(), 0.12) * 1.35
ax.axvline(0.1, color=INK2, ls="--", lw=1)
ax.text(0.102, len(labels) - 0.55, "เกณฑ์สมดุล 0.1", color=INK2, fontsize=9, va="center")
ax.set_yticks(y, labels); ax.invert_yaxis()
ax.set_xlabel("|SMD|  (ยิ่งใกล้ 0 ยิ่งเท่ากัน)")
ax.set_title("3) GA เทียบกับการสุ่มธรรมดา")
ax.legend(frameon=False, loc="lower right"); ax.grid(axis="y", visible=False)
ax.set_xlim(0, xmax)

# 4) boxplots
for k, (col, th, unit) in enumerate([("weight", "น้ำหนัก", "kg"), ("height", "ส่วนสูง", "cm"),
                                     ("age", "อายุ", "ปี"), ("bmi", "BMI", "kg/m²")]):
    ax = fig.add_subplot(gs[1, 3 * k:3 * k + 3])
    data = [t.loc[t.team == tm, col] for tm in TEAMS]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showmeans=True,
                    medianprops=dict(color=INK, lw=2),
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor=INK, markersize=7),
                    whiskerprops=dict(color=INK2), capprops=dict(color=INK2),
                    flierprops=dict(marker="o", markersize=5, markerfacecolor=MUTED, markeredgecolor="none"))
    for patch, tm in zip(bp["boxes"], TEAMS):
        patch.set_facecolor(TEAM_C[tm]); patch.set_alpha(0.85); patch.set_edgecolor("#fcfcfb")
    for i, d in enumerate(data, start=1):
        ax.text(i, ax.get_ylim()[1], f"เฉลี่ย {d.mean():.1f}\nSD {d.std():.1f}", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_xticks([1, 2], ["ทีมแดง", "ทีมฟ้า"]); ax.set_ylabel(unit)
    ax.set_title(f"4.{k+1}) {th}", pad=34); ax.grid(axis="x", visible=False)

# 5) BMI groups
ax = fig.add_subplot(gs[2, 0:4])
ctb = pd.crosstab(t.bmi_group, t.team).reindex(index=BMI_LABELS, columns=TEAMS, fill_value=0)
team_bars(ax, ctb, BMI_LABELS, ["ผอม\n<18.5", "ปกติ\n18.5–22.9", "ท้วม\n23–24.9", "อ้วน\n25–29.9", "อ้วนมาก\n30+"])
ax.set_title("5) จำนวนคนแต่ละกลุ่ม BMI (เกณฑ์เอเชีย)")

# 6) department split
ax = fig.add_subplot(gs[2, 4:12])
dep = pd.crosstab(t.department, t.team).reindex(columns=TEAMS, fill_value=0)
dep = dep.loc[dep.sum(axis=1).sort_values(ascending=False).index]
xx = np.arange(len(dep))
ax.bar(xx, dep.Red, 0.7, color=RED, label="ทีมแดง", edgecolor="#fcfcfb", linewidth=2)
ax.bar(xx, dep.Blue, 0.7, bottom=dep.Red, color=BLUE, label="ทีมฟ้า", edgecolor="#fcfcfb", linewidth=2)
for i, (r, b) in enumerate(zip(dep.Red, dep.Blue)):
    ax.text(i, r + b + 0.8, f"{r}/{b}", ha="center", color=INK, fontsize=9)
ax.set_xticks(xx, dep.index); ax.set_xlabel("หน่วยงาน (เรียงตามขนาด)  —  ตัวเลข = แดง/ฟ้า")
ax.set_ylabel("จำนวนคน"); ax.set_ylim(0, dep.sum(axis=1).max() * 1.12)
ax.set_title("6) แต่ละหน่วยงานแบ่งไปสองสี  (ทุกหน่วยงานที่มี 2 คนขึ้นไปมีทั้งสองสี; B มี 1 คน)")
ax.legend(frameon=False, loc="upper right"); ax.grid(axis="x", visible=False)

fig.suptitle("ผลการแบ่งทีมกีฬาสีด้วย Genetic Algorithm  —  แดง 179 คน / ฟ้า 179 คน",
             x=0.06, ha="left", y=0.965, fontsize=16, fontweight="bold", color=INK)
fig.savefig("ga_team_results.png", dpi=150, bbox_inches="tight")
print("saved ga_team_results.png")
