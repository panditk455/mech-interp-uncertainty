"""Two-way clustered percentile bootstrap for crossed, non-independent data.

Generalizes the examiner x cset structure from the forensic firearm examination
COMPS project (Pandit, Wang, Chang 2026 — see forensic_foundation/) to any
unit x cluster1 x cluster2 design: model x prompt, agent x task, rater x item.

Combined bootstrap follows Cameron, Gelbach & Miller (2011):
    theta*_combined = theta*_cluster1 + theta*_cluster2 - theta*_row
"""

import numpy as np
import pandas as pd

DEFAULT_METHODS = ("row", "cluster1", "cluster2", "combined")


def rate_statistic(df, outcome_col, stage1_col=None):
    """Rate of outcome_col == 1.

    If stage1_col is given, treats the data as a two-stage process — e.g.
    abstain/refuse (stage1) then, conditional on not abstaining, correct/incorrect
    (outcome) — mirroring the inconclusive/error structure in the forensic model.
    Returns {"stage1_rate": ..., "conditional_rate": ...} in that case, or
    {"rate": ...} otherwise.
    """
    if stage1_col is None:
        return {"rate": df[outcome_col].mean()}

    stage1 = df[stage1_col]
    conditional = df.loc[stage1 == 0, outcome_col]
    return {
        "stage1_rate": stage1.mean(),
        "conditional_rate": conditional.mean() if len(conditional) else np.nan,
    }


def _resample_rows(df, rng):
    positions = rng.integers(0, len(df), size=len(df))
    return df.iloc[positions]


def _resample_cluster(df, index_map, rng):
    labels = np.fromiter(index_map.keys(), dtype=object, count=len(index_map))
    sampled = rng.choice(labels, size=len(labels), replace=True)
    pos_chunks = [index_map[lab] for lab in sampled]
    return df.iloc[np.concatenate(pos_chunks)]


def bootstrap_replicates(
    df,
    method,
    B,
    cluster1_col=None,
    cluster2_col=None,
    statistic_fn=rate_statistic,
    statistic_kwargs=None,
    seed=None,
):
    """Draw B bootstrap replicates of statistic_fn(df, **statistic_kwargs).

    method: "row" resamples individual observations (ignores clustering);
    "cluster1"/"cluster2" resample whole clusters with replacement.
    """
    statistic_kwargs = statistic_kwargs or {}
    rng = np.random.default_rng(seed)

    if method == "cluster1":
        index_map = df.groupby(cluster1_col, sort=False).indices
    elif method == "cluster2":
        index_map = df.groupby(cluster2_col, sort=False).indices
    elif method != "row":
        raise ValueError(f"unknown method {method!r}")

    records = []
    for _ in range(B):
        if method == "row":
            boot_df = _resample_rows(df, rng)
        else:
            boot_df = _resample_cluster(df, index_map, rng)
        records.append(statistic_fn(boot_df, **statistic_kwargs))
    return pd.DataFrame.from_records(records)


def percentile_ci(replicates, conf_level=0.95):
    alpha = 1 - conf_level
    return {
        col: (
            np.nanquantile(replicates[col], alpha / 2),
            np.nanquantile(replicates[col], 1 - alpha / 2),
        )
        for col in replicates.columns
    }


def two_way_cluster_bootstrap(
    df,
    cluster1_col,
    cluster2_col,
    B=300,
    statistic_fn=rate_statistic,
    statistic_kwargs=None,
    conf_level=0.95,
    seed=None,
    methods=DEFAULT_METHODS,
):
    """Naive/cluster1/cluster2/combined percentile CIs for statistic_fn(df).

    Returns {method: {"observed": {...}, "ci": {stat: (lo, hi)}, "replicates": DataFrame}}.
    """
    statistic_kwargs = statistic_kwargs or {}
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=3)

    rep_row = bootstrap_replicates(df, "row", B, cluster1_col, cluster2_col, statistic_fn, statistic_kwargs, seed=seeds[0])
    rep_c1 = bootstrap_replicates(df, "cluster1", B, cluster1_col, cluster2_col, statistic_fn, statistic_kwargs, seed=seeds[1])
    rep_c2 = bootstrap_replicates(df, "cluster2", B, cluster1_col, cluster2_col, statistic_fn, statistic_kwargs, seed=seeds[2])
    rep_comb = rep_c1 + rep_c2 - rep_row

    observed = statistic_fn(df, **statistic_kwargs)
    replicate_tables = {"row": rep_row, "cluster1": rep_c1, "cluster2": rep_c2, "combined": rep_comb}

    return {
        m: {
            "observed": observed,
            "ci": percentile_ci(replicate_tables[m], conf_level),
            "replicates": replicate_tables[m],
        }
        for m in methods
    }
