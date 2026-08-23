# Tier 3 — clustered uncertainty for interpretability probes

Not yet implemented — this is the stretch goal.

Idea: train a linear probe on model activations for something safety-relevant
(e.g. "does the model represent that it's being evaluated," or the refusal
direction from Arditi et al., "Refusal in Language Models Is Mediated by a
Single Direction"). Interpretability papers routinely report a single probe
accuracy number across held-out prompts with no correction for the fact that
prompts cluster by template and models cluster by family/checkpoint.

Plan:
1. Extract activations for a set of (model, prompt) pairs using something like
   `nnsight` or `TransformerLens`.
2. Train the probe, get per-(model, prompt) correct/incorrect predictions.
3. Feed that long-format table into `clustered_bootstrap.two_way_cluster_bootstrap`
   with `cluster1_col="model"`, `cluster2_col="prompt_template"` — the exact same
   call as Tier 1, just on probe accuracy instead of eval outcome.
4. Compare the naive CI (what most papers implicitly report) against the
   combined CI, to show how much the uncertainty widens.

Needs open-weight model access and activation-extraction tooling that isn't
set up yet — treat as future work until scoped.
