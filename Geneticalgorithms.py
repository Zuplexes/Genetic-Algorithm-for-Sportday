import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. โหลดและเตรียมข้อมูล (358 คน)
df = pd.read_csv('HR.csv')
df['department'] = df['department'].str.strip()
N = len(df)

# แปลงข้อมูลเป็นเมทริกซ์มาตรฐาน (Z-score สำหรับตัวเลข และ One-hot สำหรับแผนก)
num_cols = df[['weight', 'height', 'age']].apply(lambda x: (x - x.mean()) / x.std()).values
gender_col = (df['gender'] == 'Male').values[:, None].astype(float)
dept_cols = pd.get_dummies(df['department'], dtype=float).values

# 2. Fitness Function (วัดความเท่าเทียม + กระจายทุกแผนก 50:50)
def get_fitness(pop):
    sign = pop * 2 - 1  # แปลง 0, 1 เป็น -1 (Team Pink) และ +1 (Team Red)
    gender_err = np.abs(sign @ gender_col).ravel() / N
    phys_err = np.abs(sign @ num_cols).mean(axis=1) / (N / 2)
    dept_err = (np.abs(sign @ dept_cols) - (dept_cols.sum(0) % 2)).sum(axis=1) / N
    return 100 / (1 + 10 * (2 * gender_err + phys_err + 5 * dept_err))

# 3. Genetic Algorithm (Regeneration Loop)
np.random.seed(42)
pop = np.array([np.random.permutation([0] * (N // 2) + [1] * (N // 2)) for _ in range(150)], dtype=float)
history = []

for gen in range(80):
    fit = get_fitness(pop)
    history.append(fit.max())
    elite = pop[np.argsort(fit)[-15:]]  # Elitism: เก็บตัวเก่งสุด 15 อันดับแรก
    children = []
    for _ in range(len(pop) - 15):
        p1, p2 = elite[np.random.choice(15, 2, replace=False)]  # Selection
        c = np.where(np.random.rand(N) < 0.5, p1, p2)  # Crossover

        # ซ่อมจำนวนคนให้เท่ากัน 179 : 179 เสมอ
        diff = int(c.sum()) - N // 2
        if diff != 0:
            idx = np.where(c == (1 if diff > 0 else 0))[0]
            c[np.random.choice(idx, abs(diff), replace=False)] = (0 if diff > 0 else 1)

        # Swap Mutation: สุ่มสลับคนระหว่าง Team Pink และ Team Red 1 คู่
        if np.random.rand() < 0.4:
            i1, i0 = np.random.choice(np.where(c == 1)[0]), np.random.choice(np.where(c == 0)[0])
            c[i1], c[i0] = 0, 1
        children.append(c)
    pop = np.vstack([elite, children])

# 4. แปลงป้ายชื่อเป็น Team Pink / Team Red และบันทึกไฟล์ CSV
best_chrom = pop[np.argmax(get_fitness(pop))]
df['team'] = np.where(best_chrom == 0, 'Team Pink', 'Team Red')
df.to_csv('HR_sports_day_teams.csv', index=False)

print(f"Fitness Score: Gen 0 = {history[0]:.2f} -> Gen 80 = {history[-1]:.2f}\n")
print("=== Mean & SD of Physical Attributes (น้ำหนัก, ส่วนสูง, อายุ) ===\n",
      df.groupby('team')[['weight', 'height', 'age']].agg(['mean', 'std']).round(2))
print("\n=== Gender Distribution (สัดส่วนเพศ) ===\n", pd.crosstab(df['team'], df['gender']))
print("\n=== Department Distribution (สัดส่วนแผนก) ===\n", pd.crosstab(df['department'], df['team']).T)

# 5. พล็อตกราฟแสดงผลลัพธ์ (3 กราฟ: Fitness, แผนก, และ น้ำหนัก/ส่วนสูง/อายุ - ธีมสีชมพู/แดง)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

# กราฟที่ 1: ค่า Fitness
ax[0].plot(history, lw=2.5, color='#ff69b4')
ax[0].set_title('1. GA Fitness Convergence')
ax[0].set_xlabel('Generation')
ax[0].set_ylabel('Fitness Score')

# กราฟที่ 2: การกระจายคนตามแผนก (สีชมพู และ สีแดง)
pd.crosstab(df['department'], df['team']).plot(
    kind='bar', ax=ax[1], title='2. Department Split (50:50)', color=['#ff69b4', '#e74c3c']
)
ax[1].set_ylabel('Number of Employees')

# กราฟที่ 3: เปรียบเทียบค่าเฉลี่ย น้ำหนัก ส่วนสูง อายุ
phys_means = df.groupby('team')[['weight', 'height', 'age']].mean().T
phys_means.plot(
    kind='bar', ax=ax[2], title='3. Mean Weight, Height & Age', color=['#ff69b4', '#e74c3c'], rot=0
)
ax[2].set_ylabel('Average Value')

plt.tight_layout()
plt.show()
