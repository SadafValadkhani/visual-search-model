import sys; sys.path.append("src")
from model import Config, run_agent

df = run_agent(Config(), seed=1)
print(df.head())
print(df.p_hat_hp.iloc[[0, 50, 200, -1]])