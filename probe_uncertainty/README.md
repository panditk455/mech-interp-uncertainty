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

**Result (3 models x 40 questions x 4 templates x 2 conditions, 960 rows,
leave-one-template-out CV):** 99.48% probe accuracy — the probe correctly
classifies eval-aware vs. deployment framing even for a phrasing it never
saw during training, not just memorized strings. Per model: Qwen2.5-1.5B
100%, Phi-3.5-mini 99.7%, Qwen2.5-3B 98.75%. All 5 errors (out of 960)
concentrate on one specific held-out phrasing ("Notice: you are currently
participating in a benchmark assessment...") — Qwen2.5-3B misreads the
matching *deployment*-framed variant of that template as eval-aware in all
4 of its errors, a specific, reproducible near-miss rather than noise.
Naive CI [98.96%, 99.90%] (width 0.94pts) vs. combined CI [98.65%, 100.31%]
(width 1.67pts) — ~1.8x wider, consistent with Phase A and B. The combined
upper bound spilling past 100% is a known artifact of the Cameron-Gelbach-
Miller additive combination near a boundary with low variance, not a data
error. Caveat: the 4 phrasings per condition, while independently worded,
still share theme-consistent vocabulary within a class ("evaluate/test/
benchmark/scored" vs. "helpful/user/chatting/assistant") — this is a real
step up from literal-string memorization, but not proof of generalization
to a phrasing sharing no vocabulary at all. Data in
`probe_uncertainty/data/probe_results.csv`, summary in
`probe_uncertainty/data/probe_bootstrap_result.txt`.

**Follow-up check — is 99.48% actually a length confound?** The eval-aware
phrasings run consistently longer than the deployment ones (8-19 tokens,
every template, both tokenizers checked), and the probe reads a
last-token activation, so sequence length was a plausible alternative
explanation. [`length_confound_check.py`](length_confound_check.py) tests a
length-only linear control under the identical leave-one-template-out CV:
**52.8% (Qwen2.5-1.5B) and 52.5% (Phi-3.5-mini)** — indistinguishable from
the 50% chance floor. Per-question variation (2-13 answer choices) swamps
the small, consistent length gap, ruling out length as the driver of the
real probe's accuracy. Runs on CPU, no GPU needed:

```bash
python -m probe_uncertainty.length_confound_check
```

Result in `probe_uncertainty/data/length_confound_result.txt`.

Status: done, with a real result and a ruled-out confound, matching Phase A
and B's format.
