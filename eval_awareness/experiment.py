"""Does a model behave differently when it can tell it's being evaluated?

Not yet run — this is a skeleton for a live experiment. Fill in `run_condition`
once you've picked models, an inference API, and a benchmark.

Design:
  1. Run the same set of models on the same benchmark twice: once with framing
     that discloses it's an evaluation (e.g. an eval-canary system prompt), once
     framed as ordinary deployment use.
  2. Record, per (model, prompt): did it abstain/refuse (stage1), and if not,
     was it correct/safe (outcome) — the same two-stage shape as the forensic
     inconclusive-then-error model.
  3. Fit clustered_bootstrap.rate_statistic with stage1_col set, clustered by
     model and by prompt, separately for each framing condition, and compare.

This mirrors sandbagging / evaluation-awareness findings (e.g. Greenblatt et al.,
"Alignment Faking in Large Language Models"; Apollo Research's in-context
scheming work) — verify exact citations before writing these up formally.
"""

import pandas as pd

from clustered_bootstrap import rate_statistic, two_way_cluster_bootstrap


def run_condition(models, prompts, eval_aware: bool):
    """Run `models` on `prompts` under one framing condition.

    Must return a long-format DataFrame with columns: model, prompt, abstained,
    outcome. Not implemented — wire this up to your inference API of choice.
    """
    raise NotImplementedError("Wire up model calls here before running this experiment.")


def compare_conditions(models, prompts, B=1000, seed=42):
    aware_df = run_condition(models, prompts, eval_aware=True)
    naive_df = run_condition(models, prompts, eval_aware=False)

    results = {}
    for label, df in (("eval_aware", aware_df), ("deployment_framed", naive_df)):
        results[label] = two_way_cluster_bootstrap(
            df,
            cluster1_col="model",
            cluster2_col="prompt",
            B=B,
            statistic_fn=lambda d: rate_statistic(d, outcome_col="outcome", stage1_col="abstained"),
            seed=seed,
        )
    return results


if __name__ == "__main__":
    raise SystemExit(
        "This is a skeleton — implement run_condition() with real model calls, "
        "then call compare_conditions() with your model/prompt lists."
    )
