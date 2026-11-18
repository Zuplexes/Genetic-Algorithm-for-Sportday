import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('HR.csv')
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
n = len(df)

g_dum, d_dum = pd.get_dummies(df['gender']).values, pd.get_dummies(df['department']).values
phys = df[['weight', 'height', 'age', 'bmi']].values


def calc_fitness(chrom):
    mask_a, mask_b = (chrom == 0), (chrom == 1)
    if mask_a.sum() == 0 or mask_b.sum() == 0: return 1e6

    sz_p = abs(mask_a.sum() - mask_b.sum()) * 10
    dp_p = np.abs(d_dum[mask_a].sum(axis=0) - d_dum[mask_b].sum(axis=0)).sum() * 5
    gd_p = np.abs(g_dum[mask_a].sum(axis=0) - g_dum[mask_b].sum(axis=0)).sum() * 5
    ph_p = np.abs(phys[mask_a].mean(axis=0) - phys[mask_b].mean(axis=0)).sum() * 2

    return sz_p + dp_p + gd_p + ph_p


pop_size, generations = 100, 50
pop = [np.random.permutation([0] * (n // 2) + [1] * (n - n // 2)) for _ in range(pop_size)]
best_hist, best_chrom, best_fit = [], None, float('inf')

for _ in range(generations):
    scores = [calc_fitness(c) for c in pop]
    if min(scores) < best_fit:
        best_fit, best_chrom = min(scores), pop[np.argmin(scores)].copy()
    best_hist.append(best_fit)

    new_pop = [best_chrom]
    for _ in range(pop_size - 1):
        p1 = pop[min(np.random.choice(pop_size, 2), key=lambda x: scores[x])]
        p2 = pop[min(np.random.choice(pop_size, 2), key=lambda x: scores[x])]

        child = np.where(np.random.rand(n) > 0.5, p1, p2)
        if np.random.rand() < 0.05:
            a, b = np.random.choice(np.where(child == 0)[0]), np.random.choice(np.where(child == 1)[0])
            child[a], child[b] = 1, 0
        new_pop.append(child)
    pop = new_pop

df['Team'] = np.where(best_chrom == 0, 'Team A', 'Team B')

with pd.ExcelWriter('Sports_Day_Teams.xlsx') as writer:
    df.to_excel(writer, sheet_name='All Employees', index=False)
    df.groupby('Team').agg(
        Total=('Team', 'count'),
        Male=('gender', lambda x: (x == 'Male').sum()),
        Female=('gender', lambda x: (x == 'Female').sum()),
        Age_Avg=('age', 'mean'),
        Weight_Avg=('weight', 'mean'),
        Height_Avg=('height', 'mean'),
        BMI_Avg=('bmi', 'mean')
    ).round(1).to_excel(writer, sheet_name='Summary')

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
ax = axes.flatten()
team_colors = ['#4361EE', '#F72585']

ax[0].plot(best_hist, color='#06D6A0', lw=3)
ax[0].set_title('Fitness Score (Lower is Better)')
ax[0].set_xlabel('Generation')
ax[0].set_ylabel('Penalty Score')

sns.countplot(data=df, x='department', hue='Team', ax=ax[1], palette=team_colors)
ax[1].set_title('Department Distribution')
ax[1].set_xlabel('Department')
ax[1].set_ylabel('Count')
for container in ax[1].containers:
    ax[1].bar_label(container, fmt='%d', padding=3, fontsize=9)

sns.countplot(data=df, x='gender', hue='Team', ax=ax[2], palette=team_colors)
ax[2].set_title('Gender Distribution')
ax[2].set_xlabel('Gender')
ax[2].set_ylabel('Count')
for container in ax[2].containers:
    ax[2].bar_label(container, fmt='%d', padding=3, fontsize=9)

bmi_avg = df.groupby('Team')['bmi'].mean().reset_index()
sns.barplot(data=bmi_avg, x='Team', y='bmi', ax=ax[3], palette=team_colors)
ax[3].set_title('Average BMI')
ax[3].set_xlabel('Team')
ax[3].set_ylabel('BMI')
ax[3].set_ylim(20, 30)
for container in ax[3].containers:
    ax[3].bar_label(container, fmt='%.1f', padding=3, fontsize=9)

sns.despine()
plt.tight_layout()
plt.show()