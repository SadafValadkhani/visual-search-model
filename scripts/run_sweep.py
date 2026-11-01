import os
import numpy as np
import pandas as pd
import sys; sys.path.append("src")
from model import Config, run_agent

os.makedirs("results", exist_ok=True)

alphas = [0.0, 0.05, 0.1, 0.15, 0.2]
betas = [0.0, 0.5, 1.0, 1.5, 2.0]
n_agents = 10

rows = []
for a in alphas:
    for b in betas:
        caps_hp = []
        for agent in range(n_agents):
            cfg = Config(alpha=a, beta=b, n_trials=800)
            df = run_agent(cfg, seed=agent)
            last = df[df["trial"] >= 720]                   # final block
            hp = last[last["distractor"] == 0]["captured"].mean()
            caps_hp.append(hp)
        rows.append(dict(alpha=a, beta=b, hp_capture=np.mean(caps_hp)))
        print("done alpha", a, "beta", b)

pd.DataFrame(rows).to_csv("results/sweep.csv", index=False)
print("Saved results/sweep.csv")