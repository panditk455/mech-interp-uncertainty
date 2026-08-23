"""Sanity check for the Python port: a synthetic two-way clustered binary outcome
(cluster1 = e.g. model/rater, cluster2 = e.g. prompt/item) should reproduce the same
qualitative result as the R forensic simulation in forensic_foundation/ — the naive
bootstrap under-covers the true rate, and the two-way combined bootstrap does not.

Run: python -m examples.synthetic_validation
"""

import numpy as np
import pandas as pd

from clustered_bootstrap import coverage_simulation, rate_statistic

N_CLUSTER1 = 30  # e.g. models being evaluated
N_CLUSTER2 = 50  # e.g. prompts / tasks
SD_CLUSTER1 = 1.0
SD_CLUSTER2 = 0.8
INTERCEPT = -0.5


def _inv_logit(x):
    return 1 / (1 + np.exp(-x))


def true_rate():
    rng = np.random.default_rng(0)
    n_mc = 200_000
    u = rng.normal(0, SD_CLUSTER1, n_mc)
    v = rng.normal(0, SD_CLUSTER2, n_mc)
    p = _inv_logit(INTERCEPT + u + v)
    return {"rate": p.mean()}


def generate(seed):
    rng = np.random.default_rng(seed)
    u = rng.normal(0, SD_CLUSTER1, N_CLUSTER1)
    v = rng.normal(0, SD_CLUSTER2, N_CLUSTER2)

    eta = INTERCEPT + u[:, None] + v[None, :]
    p = _inv_logit(eta)
    outcome = rng.binomial(1, p)

    c1_idx, c2_idx = np.meshgrid(np.arange(N_CLUSTER1), np.arange(N_CLUSTER2), indexing="ij")
    return pd.DataFrame(
        {
            "cluster1": [f"unit1_{i}" for i in c1_idx.ravel()],
            "cluster2": [f"unit2_{j}" for j in c2_idx.ravel()],
            "outcome": outcome.ravel(),
        }
    )


def statistic(df):
    return rate_statistic(df, outcome_col="outcome")


if __name__ == "__main__":
    summary, _ = coverage_simulation(
        data_generator=generate,
        true_value_fn=true_rate,
        cluster1_col="cluster1",
        cluster2_col="cluster2",
        statistic_fn=statistic,
        M=100,
        B=200,
        seed=42,
    )
    print(summary.to_string(index=False))
