import os
import numpy as np
import pandas as pd
import sys; sys.path.append("src")
from model import Config, run_agent

os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(2026)   # fixed seed = reproducible

all_data = []
for agent in range(50):
    cfg = Config(
        alpha=max(0.01, rng.normal(0.1, 0.03)),   # individual differences
        beta=max(0.1, rng.normal(1.0, 0.2)),
    )
    df = run_agent(cfg, seed=agent)
    df["agent"] = agent
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)
data.to_csv("results/simulated_data.csv", index=False)
print("Saved", len(data), "trials")