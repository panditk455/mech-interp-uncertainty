"""Tier 1: apply the two-way clustered bootstrap to a real model x prompt eval matrix.

Expected input: a long-format CSV with one row per (model, prompt) observation:

    model,prompt,outcome
    gpt-x,jailbreak_001,1
    gpt-x,jailbreak_002,0
    claude-y,jailbreak_001,0
    ...

`outcome` is a binary column — e.g. attack_success, hallucinated, unsafe_response.
model and prompt are the two crossed clustering factors (models vary in how they
score in general; prompts vary in how likely they are to trigger the behavior,
regardless of model).

Usage:
    python -m ai_safety_extension.clustered_bootstrap_evals data.csv \\
        --model-col model --prompt-col prompt --outcome-col outcome
"""

import argparse

import pandas as pd

from clustered_bootstrap import rate_statistic, two_way_cluster_bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="Long-format CSV: one row per (model, prompt) observation")
    parser.add_argument("--model-col", default="model")
    parser.add_argument("--prompt-col", default="prompt")
    parser.add_argument("--outcome-col", default="outcome")
    parser.add_argument("--stage1-col", default=None, help="Optional: e.g. a 'refused' column for two-stage refuse-then-error analysis")
    parser.add_argument("--B", type=int, default=1000, help="Bootstrap resamples per method")
    parser.add_argument("--conf-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    def statistic(d):
        return rate_statistic(d, outcome_col=args.outcome_col, stage1_col=args.stage1_col)

    result = two_way_cluster_bootstrap(
        df,
        cluster1_col=args.model_col,
        cluster2_col=args.prompt_col,
        B=args.B,
        statistic_fn=statistic,
        conf_level=args.conf_level,
        seed=args.seed,
    )

    labels = {
        "row": "naive",
        "cluster1": f"{args.model_col}-cluster",
        "cluster2": f"{args.prompt_col}-cluster",
        "combined": "combined",
    }

    print(f"n_obs={len(df)}  n_models={df[args.model_col].nunique()}  n_prompts={df[args.prompt_col].nunique()}\n")
    print(f"{'method':<18} {'statistic':<18} {'observed':>10} {'ci_lo':>10} {'ci_hi':>10} {'width':>10}")
    for method in ("row", "cluster1", "cluster2", "combined"):
        observed = result[method]["observed"]
        for stat_name, (lo, hi) in result[method]["ci"].items():
            print(f"{labels[method]:<18} {stat_name:<18} {observed[stat_name]:>10.4f} {lo:>10.4f} {hi:>10.4f} {hi - lo:>10.4f}")


if __name__ == "__main__":
    main()
