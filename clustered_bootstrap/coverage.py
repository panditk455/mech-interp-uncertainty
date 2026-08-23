"""Monte Carlo coverage simulation: repeatedly generate data from a known
process, bootstrap CIs by each method, and check whether they cover the known
true value. Generalizes Sim_boot_cov_all from forensic_foundation/simulation_combined/.
"""

import pandas as pd

from .bootstrap import DEFAULT_METHODS, rate_statistic, two_way_cluster_bootstrap


def coverage_simulation(
    data_generator,
    true_value_fn,
    cluster1_col,
    cluster2_col,
    statistic_fn=rate_statistic,
    statistic_kwargs=None,
    M=200,
    B=300,
    conf_level=0.95,
    seed=42,
    methods=DEFAULT_METHODS,
):
    """
    data_generator(seed) -> DataFrame with cluster1_col, cluster2_col, and whatever
        columns statistic_fn needs.
    true_value_fn() -> dict with the same keys statistic_fn returns, giving the
        ground-truth value each key should converge to.

    Returns (summary_df, per_method_repetition_tables).
    """
    statistic_kwargs = statistic_kwargs or {}
    true_value = true_value_fn()
    records = {m: [] for m in methods}

    for rep in range(M):
        rep_seed = seed + rep
        df = data_generator(rep_seed)
        result = two_way_cluster_bootstrap(
            df,
            cluster1_col,
            cluster2_col,
            B=B,
            statistic_fn=statistic_fn,
            statistic_kwargs=statistic_kwargs,
            conf_level=conf_level,
            seed=rep_seed,
            methods=methods,
        )
        for m in methods:
            row = {"rep": rep}
            for key, (lo, hi) in result[m]["ci"].items():
                truth = true_value[key]
                row[f"{key}_lo"] = lo
                row[f"{key}_hi"] = hi
                row[f"{key}_covered"] = truth >= lo and truth <= hi
                row[f"{key}_width"] = hi - lo
            records[m].append(row)

    per_method = {m: pd.DataFrame(records[m]) for m in methods}

    summary_rows = []
    for m in methods:
        tbl = per_method[m]
        for key in true_value:
            summary_rows.append(
                {
                    "method": m,
                    "statistic": key,
                    "true_value": true_value[key],
                    "coverage_rate": tbl[f"{key}_covered"].mean(),
                    "mean_ci_width": tbl[f"{key}_width"].mean(),
                    "nominal_coverage": conf_level,
                    "M": M,
                    "B": B,
                }
            )
    return pd.DataFrame(summary_rows), per_method
