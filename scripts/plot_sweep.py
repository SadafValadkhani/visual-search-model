import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sweep = pd.read_csv("results/sweep.csv")
table = sweep.pivot(index="beta", columns="alpha", values="hp_capture")

sns.heatmap(table, annot=True, fmt=".2f", cmap="viridis_r")
plt.title("HP distractor capture rate, final block")
plt.gca().invert_yaxis()
plt.savefig("results/sweep_heatmap.png", dpi=200)