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

Plan, now concrete: [`probe_uncertainty_colab.ipynb`](probe_uncertainty_colab.ipynb) —
run on Colab's free T4 GPU (this repo's dev machine has 8GB RAM, too tight
for real activation extraction).

1. Builds the same eval-disclosing vs deployment-framed system-prompt
   contrast as `eval_awareness/experiment.py` (Phase B), on the same
   TruthfulQA questions, but instead of reading the model's final answer
   letter, extracts its last-token hidden state at a middle layer for 3
   small ungated instruct models (Qwen2.5-1.5B, Qwen2.5-3B, Phi-3.5-mini).
   Each framing has 4 independently-worded phrasings, not one fixed string —
   see the revision note below for why that matters.
2. Trains a logistic-regression probe per model to classify eval-aware vs
   deployment framing from that activation — "does the model represent that
   it's being evaluated" — using **leave-one-template-out** CV: the probe
   never sees the held-out phrasing during training, so accuracy reflects a
   generalizable direction rather than memorized surface strings.
3. Builds a long-format `(model, prompt, outcome)` table and feeds it into
   `clustered_bootstrap.two_way_cluster_bootstrap` with
   `cluster1_col="model"`, `cluster2_col="prompt"` — the exact same call as
   `benchmark_uncertainty/` and `eval_awareness/`, just on probe accuracy.
4. Compares the naive CI (what most papers implicitly report) against the
   combined CI, to show how much the uncertainty widens.

**Revision note:** the first Colab run used one fixed string per condition
and got 100% probe accuracy on every single one of 240 rows across all 3
models. That's not a result — it's a sign the task was trivially separable
(the probe can just detect "which literal string was in the context,"
independent of any real represented concept). This is precisely the
template-clustering problem this phase exists to surface. Fixed by adding
template diversity and leave-one-template-out CV (above); the CV logic was
sanity-checked against synthetic data with a known template confound before
re-running on GPU.

Status: revised notebook written, CV logic verified on synthetic data, not
yet re-run on Colab.
