# probe_uncertainty

How much do we actually know about what a probe found?

Idea: train a linear probe on model activations for something safety-relevant
(e.g. "does the model represent that it's being evaluated," or the refusal
direction from Arditi et al., "Refusal in Language Models Is Mediated by a
Single Direction"). Interpretability papers routinely report a single probe
accuracy number across held-out prompts with no correction for the fact that
prompts cluster by template and models cluster by family/checkpoint — this is
the most statistically underserved corner of the three directions in this repo,
which is exactly what makes it interesting.

Plan:
1. Extract activations for a set of (model, prompt) pairs using something like
   `nnsight` or `TransformerLens`.
2. Train the probe, get per-(model, prompt) correct/incorrect predictions.
3. Feed that long-format table into `clustered_bootstrap.two_way_cluster_bootstrap`
   with `cluster1_col="model"`, `cluster2_col="prompt_template"` — the exact same
   call as `benchmark_uncertainty/`, just on probe accuracy instead of eval outcome.
4. Compare the naive CI (what most papers implicitly report) against the
   combined CI, to show how much the uncertainty widens.

Needs open-weight model access and activation-extraction tooling that isn't
set up yet — early-stage, but this is the direction with the most novel payoff.
