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

1. Reuses the exact eval-disclosing vs deployment-framed system prompts from
   `eval_awareness/experiment.py` (Phase B), on the same TruthfulQA questions,
   but instead of reading the model's final answer letter, extracts its
   last-token hidden state at a middle layer for 3 small ungated instruct
   models (Qwen2.5-1.5B, Qwen2.5-3B, Phi-3.5-mini).
2. Trains a logistic-regression probe per model to classify eval-aware vs
   deployment framing from that activation — "does the model represent that
   it's being evaluated" — using group-K-fold CV grouped by question, so a
   question's activations never leak across train/test.
3. Builds a long-format `(model, prompt, outcome)` table and feeds it into
   `clustered_bootstrap.two_way_cluster_bootstrap` with
   `cluster1_col="model"`, `cluster2_col="prompt"` — the exact same call as
   `benchmark_uncertainty/` and `eval_awareness/`, just on probe accuracy.
4. Compares the naive CI (what most papers implicitly report) against the
   combined CI, to show how much the uncertainty widens.

Status: notebook is written and reasoned through but not yet run end-to-end —
next step is opening it in Colab and working through whatever breaks.
