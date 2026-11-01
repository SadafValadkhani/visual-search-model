import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv("results/simulated_data.csv")

# label trials: 800 trials -> 10 blocks (0-9)
data["block"] = data["trial"] // 80
data["dist_type"] = (data["distractor"] == 0).map({True: "HP", False: "LP"})
data["captured"] = data["captured"].astype(int)

# 1. capture rate by block
cap = data.groupby(["block", "dist_type"])["captured"].mean().unstack()
print("\nCAPTURE RATE BY BLOCK")
print(cap)
cap.plot(marker="o")
plt.ylabel("Capture rate")
plt.xlabel("Block")
plt.title("Learning agents")
plt.ylim(0, 1)
plt.savefig("results/capture_by_block.png", dpi=200)

# 2. mixed model on RT
print("\nMIXED MODEL: RT")
m = smf.mixedlm("rt ~ C(dist_type) * block", data, groups=data["agent"]).fit()
print(m.summary())

# 3. logistic model on capture (GEE)
print("\nGEE: CAPTURE")
g = smf.gee("captured ~ C(dist_type) * block", groups="agent",
            data=data, family=sm.families.Binomial()).fit()
print(g.summary())