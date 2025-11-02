# Functional demo for the "VOICE + boundary + relaxation" loop

import numpy as np
import pandas as pd
import math
import random
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


# --- Core components ---

@dataclass
class Config:
    H: int = 300                 # horizon
    t0: int = 120                # center of boundary window
    tau: float = 30.0            # relaxation / window half-width
    eps_inf: float = 1.0         # asymptote for epsilon
    actions: Tuple[str, str] = ("A", "B")
    alpha_beta_prior: float = 1.0  # Beta prior alpha=beta=1 (uniform)
    voice_noise_scale: float = 1.0 # multiplies (1 - k) inside flip probability
    seed: int = 7

random.seed(7)
np.random.seed(7)


def bernoulli(p: float) -> int:
    return 1 if random.random() < p else 0


def flip_against(a_star: int, k: float, noise_scale: float, n_actions: int) -> int:
    """
    VOICE diagonalization guard: with probability q = noise_scale*(1-k),
    choose an action different from a_star (uniform among others).
    Otherwise keep a_star.
    """
    q = max(0.0, min(1.0, noise_scale * (1.0 - k)))
    if n_actions == 1:
        return a_star
    if random.random() < q:
        # pick any action != a_star
        others = [i for i in range(n_actions) if i != a_star]
        return random.choice(others)
    return a_star


class BetaBandit:
    """
    Simple Bayesian Bernoulli bandit learner for P_t(a).
    """
    def __init__(self, n_actions: int, prior: float):
        self.alpha = np.ones(n_actions) * prior
        self.beta = np.ones(n_actions) * prior

    def probs(self) -> np.ndarray:
        # posterior means (expected reward)
        return self.alpha / (self.alpha + self.beta)

    def update(self, action: int, reward: int):
        self.alpha[action] += reward
        self.beta[action] += 1 - reward


class Environment:
    """
    Two-action environment with drifting ground-truth probabilities.
    The optimal action changes a few times across H via a slow sine drift.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def p_reward(self, t: int) -> np.ndarray:
        # drift between actions using a smooth function; keep within (0.05, 0.95)
        base = 0.7 + 0.2 * math.sin(2 * math.pi * (t / (self.cfg.H * 0.8)))
        # action 0 follows base, action 1 is its complement-ish
        p0 = np.clip(base, 0.05, 0.95)
        p1 = np.clip(1.0 - base + 0.1 * math.sin(2 * math.pi * (t / (self.cfg.H * 0.3))), 0.05, 0.95)
        return np.array([p0, p1])

    def optimal_action(self, t: int) -> int:
        probs = self.p_reward(t)
        return int(np.argmax(probs))

    def step(self, t: int, action: int) -> int:
        probs = self.p_reward(t)
        return bernoulli(probs[action])


def run_demo(cfg: Config) -> pd.DataFrame:
    env = Environment(cfg)
    learner = BetaBandit(n_actions=len(cfg.actions), prior=cfg.alpha_beta_prior)

    records = []
    k_hat = 0.0
    eps = 0.0

    # EMA decay for k_hat chosen to mirror relaxation scale
    ema_decay = math.exp(-1.0 / max(cfg.tau, 1.0))

    for t in range(cfg.H + 1):
        # boundary indicator
        C = 1 if abs(t - cfg.t0) <= cfg.tau else 0
        # filtered estimate of window occupancy
        k_hat = ema_decay * k_hat + (1 - ema_decay) * C

        # relaxation variable
        eps = cfg.eps_inf * (1.0 - math.exp(-t / max(cfg.tau, 1e-9)))

        # predictive distribution over actions from Bayesian bandit
        P = learner.probs()
        a_star = int(np.argmax(P))
        a_voice = flip_against(
            a_star=a_star,
            k=k_hat,
            noise_scale=cfg.voice_noise_scale,
            n_actions=len(cfg.actions),
        )

        # Bernoulli-time gating: act with prob k_hat, else just observe baseline (no update)
        acted = bernoulli(k_hat)
        reward = None
        opt = env.optimal_action(t)
        if acted:
            reward = env.step(t, a_voice)
            learner.update(a_voice, reward)

        # For passive steps, still "observe" non-parametrically by logging the optimal action and env probs
        env_probs = env.p_reward(t)

        records.append({
            "t": t,
            "C": C,
            "k_hat": k_hat,
            "epsilon": eps,
            "a_star": a_star,
            "a_voice": a_voice,
            "acted": acted,
            "reward": reward if reward is not None else np.nan,
            "env_p_A": env_probs[0],
            "env_p_B": env_probs[1],
            "optimal": opt,
        })

    df = pd.DataFrame.from_records(records)
    # compute cumulative metrics
    df["cum_reward"] = df["reward"].fillna(0).cumsum()
    df["acts"] = df["acted"].cumsum()
    df["switches"] = (df["a_voice"].diff().fillna(0) != 0).astype(int).cumsum()
    return df


cfg = Config()
df = run_demo(cfg)

# Show the main time series one chart at a time
plt.figure()
plt.plot(df["t"], df["epsilon"])
plt.title("Relaxation ε(t)")
plt.xlabel("t")
plt.ylabel("epsilon")
plt.show()

plt.figure()
plt.plot(df["t"], df["k_hat"])
plt.title("Estimated coherence k(t)")
plt.xlabel("t")
plt.ylabel("k_hat")
plt.show()

# Actions vs optimal
plt.figure()
plt.step(df["t"], df["optimal"], where="post", label="optimal")
plt.step(df["t"], df["a_voice"], where="post", label="chosen (VOICE)")
plt.title("Actions: optimal vs chosen")
plt.xlabel("t")
plt.ylabel("action index")
plt.legend()
plt.show()

# Rewards & gating
plt.figure()
plt.plot(df["t"], df["cum_reward"])
plt.title("Cumulative reward (only when acted)")
plt.xlabel("t")
plt.ylabel("cum_reward")
plt.show()

# Basic summary table for quick inspection
summary = pd.DataFrame({
    "Total steps": [len(df)],
    "Acts taken": [int(df["acts"].iloc[-1])],
    "Mask sparsity (acts / steps)": [float(df["acts"].iloc[-1]) / len(df)],
    "Final cumulative reward": [float(df["cum_reward"].iloc[-1])],
    "Switches in chosen action": [int(df["switches"].iloc[-1])],
})

print("VOICE_boundary_relaxation_summary", summary)

# Save the raw trajectory for further analysis
csv_path = "/mnt/data/voice_boundary_relaxation_run.csv"
df.to_csv(csv_path, index=False)


