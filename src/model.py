from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class Config:
    n_locations: int = 6
    p_hp: float = 0.7
    hp_loc: int = 0
    n_trials: int = 800
    alpha: float = 0.1        # delta-rule learning rate
    beta: float = 1.0         # suppression strength
    s_target: float = 1.0
    s_distractor: float = 1.2
    noise_sd: float = 0.3
    cost_per_item: float = 50.0   # ms
    capture_cost: float = 0.0     # extra ms, optional
    base_rt: float = 300.0

def sample_distractor_loc(cfg, rng):
    if rng.random() < cfg.p_hp:
        return cfg.hp_loc
    others = [i for i in range(cfg.n_locations) if i != cfg.hp_loc]
    return rng.choice(others)

def run_agent(cfg, seed, learner="delta"):
    rng = np.random.default_rng(seed)
    n = cfg.n_locations
    p_hat = np.full(n, 1 / n)
    counts = np.zeros(n)
    rows = []

    for t in range(cfg.n_trials):
        d_loc = sample_distractor_loc(cfg, rng)
        t_loc = rng.choice([i for i in range(n) if i != d_loc])

        # priority map
        sal = np.zeros(n)
        sal[t_loc] = cfg.s_target
        sal[d_loc] = cfg.s_distractor
        priority = sal - cfg.beta * p_hat + rng.normal(0, cfg.noise_sd, n)

        # serial search in descending priority
        order = np.argsort(-priority)
        n_inspected = int(np.where(order == t_loc)[0][0]) + 1
        captured = order[0] == d_loc
        rt = (cfg.base_rt + n_inspected * cfg.cost_per_item
              + (cfg.capture_cost if captured else 0))

        rows.append(dict(seed=seed, trial=t, target=t_loc, distractor=d_loc,
                         first=order[0], captured=captured, rt=rt,
                         p_hat_hp=p_hat[cfg.hp_loc]))

        # learning
        obs = np.zeros(n); obs[d_loc] = 1
        if learner == "delta":
            p_hat += cfg.alpha * (obs - p_hat)
        else:  # Dirichlet counting
            counts += obs
            p_hat = (counts + 1) / (counts.sum() + n)

    return pd.DataFrame(rows)