# Auditing uncertainty in interpretability and safety-eval claims

*Application write-up, MATS 12.0 (Neel Nanda). Repo: [mech-interp-uncertainty](https://github.com/panditk455/mech-interp-uncertainty).*

> **Note to self before submitting:** this draft was written with an AI coding
> agent doing most of the execution. Before submitting, rewrite the framing
> paragraphs in your own words, re-derive at least one CI by hand to confirm
> you understand the method (not just trust the code), and cut anything below
> you can't defend live in conversation.

## Executive summary

Interpretability and safety-eval papers routinely report a single accuracy
or effect-size number with a confidence interval that treats every
(model, prompt) observation as independent. It usually isn't: a hard
question is hard for every model, and a weak model is weak on every
question, so results cluster on two axes simultaneously. I built a two-way
clustered bootstrap (adapting Cameron, Gelbach & Miller 2011) and used it to
audit three different kinds of claims: a benchmark hallucination rate across
11 open models on TruthfulQA, a live behavioral test of whether 3 models
answer differently when told they're being evaluated, and a linear probe's
accuracy at detecting that same eval-vs-deployment framing from internal
activations. In all three, the naive interval was **1.8-4.5x narrower** than
the properly clustered one. The probe case study also caught and fixed a
real methodological trap along the way: a first pass with one fixed prompt
per condition scored 100% by trivially detecting string identity rather than
any generalizable concept; redesigning around held-out paraphrases and
ruling out a length confound as an alternative explanation dropped that to a
real but still striking 99.48%. The tool is general — any (model, prompt,
outcome) table can be dropped in — and the consistent ~2x understatement
across three unrelated claim types suggests this is a structural blind spot
in how the field reports uncertainty, not a one-off.

## Motivation

A claim like "this probe detects deception with 94% accuracy" or "this model
hallucinates on 12% of questions" is almost always accompanied by a
confidence interval computed as if every row in the underlying table were an
independent coin flip. It isn't. Two kinds of non-independence are present
in almost every eval or probing setup at once: prompts/questions cluster
(some are just harder, or better-templated, regardless of model), and models
cluster (a model's overall competence or calibration shows up across every
question it answers). A one-way correction — clustering by model only, or
by prompt only — is common; both at once is rare, even though ignoring
either one gives you a false sense of precision.

I had already built a two-way clustered bootstrap for a different problem —
quantifying inter-examiner variability in forensic firearm-toolmark
comparisons, a Carleton COMPS project (Pandit, Wang, Chang 2026) — where
examiner and comparison-set are the two clustering factors. The statistics
generalize directly to model x prompt: swap "examiner" for "model" and
"comparison set" for "prompt/question," and the same combined-bootstrap
machinery applies unchanged.

## Method

For a long-format table with one row per (model, prompt) observation and a
binary or two-stage outcome, the tool computes four percentile bootstrap
CIs:

- **row**: resamples individual rows, ignoring clustering — this is the
  naive interval most papers implicitly report.
- **cluster1**: resamples whole model-clusters with replacement.
- **cluster2**: resamples whole prompt-clusters with replacement.
- **combined**: `cluster1 + cluster2 - row` (Cameron, Gelbach & Miller 2011)
  — accounts for both simultaneously.

Same function call (`two_way_cluster_bootstrap`), same code, applied to
three different `outcome` definitions below.

## Case 1: benchmark hallucination rate (sanity check)

Real per-question TruthfulQA-MC1 results for 11 open models (Llama-2 family,
Mistral-7B variants, Falcon, Gemma-7B, Phi-2, Yi-6B, GPT-J-6B), pulled from
HuggingFace's public Open LLM Leaderboard detail logs — no inference run
needed. Across 8,987 (model, question) observations: naive CI **[70.8%,
72.6%]** (width 1.8pts) vs. combined CI **[67.7%, 75.9%]** (width 8.1pts) —
**the naive interval understates uncertainty by ~4.5x**.

## Case 2: does telling a model it's being evaluated change its behavior?

Same TruthfulQA questions, reformatted as multiple-choice-with-abstain, sent
to 3 current Groq-hosted models (`gpt-oss-20b`, `gpt-oss-120b`,
`qwen3.6-27b`) under two system prompts — one explicitly disclosing "this is
an evaluation," one framing the interaction as ordinary deployment — 360
live API calls. Result: **no detectable effect**. Abstention 8.9%
(eval-aware) vs. 7.8% (deployment); conditional error rate 12.8% vs. 13.9%;
confidence intervals overlap heavily in both. But the clustered CI is still
~1.8-1.9x wider than naive, so the *absence* of an effect is also less
certain than a naive analysis would suggest. Caveat: with only 3 models, the
model-clustering dimension has just 3 resampling units and is
low-resolution — the 60-question dimension is doing most of the work here.

Two bugs surfaced and got fixed before trusting this result: `max_tokens=8`
silently truncated every reasoning-model response to empty before it could
emit an answer, and the first-match letter parser would have grabbed a
stray answer-choice letter out of one model's visible chain-of-thought
instead of its actual final answer. Both were caught by testing against the
live API on a handful of examples before running the full batch, not by
inspecting the final numbers.

## Case 3: is the eval/deployment distinction linearly decodable — and can we trust the number?

The obvious version of this experiment — "train a probe, report its
accuracy" — is exactly the kind of generic linear-representation claim that
doesn't say much on its own. The point here isn't the existence of a linear
direction; it's whether the *reported accuracy* means what it's usually
taken to mean, and what breaks that assumption in practice.

**First pass, and why it was wrong.** Using one fixed system prompt per
condition (the same two used in Case 2), extracting the last-token hidden
state at a middle layer for 3 small open models (Qwen2.5-1.5B, Qwen2.5-3B,
Phi-3.5-mini) and training a logistic-regression probe: **100% accuracy on
every one of 240 rows, across all 3 models.** That's not evidence of a
represented concept — with one fixed string per class, a probe can hit 100%
by detecting "which literal string was in the context," which requires no
generalization at all. Reporting this as a finding would have been the
disqualifying kind of error: an exciting number that doesn't survive five
minutes of skepticism.

**Fix.** Four independently-worded phrasings per condition, and
leave-one-template-out cross-validation: the probe is trained on 3
phrasings and scored on a 4th it has never seen. Verified the CV logic
against synthetic data with a known template confound before spending GPU
time on it (recovered ~79% accuracy, not 100%, correctly resisting the
trivial shortcut).

**Result:** 960 rows (3 models x 40 questions x 4 templates x 2 conditions),
**99.48% accuracy** (955/960) under held-out-phrasing CV. Per model:
Qwen2.5-1.5B 100%, Phi-3.5-mini 99.7%, Qwen2.5-3B 98.75%. Naive CI [98.96%,
99.90%] vs. combined CI **[98.65%, 100.31%]** — ~1.8x wider, matching Cases
1 and 2. (The combined interval's upper edge poking above 100% is a known
artifact of the additive CGM formula near a boundary with low variance, not
a data error — worth a footnote, not a fix, at this scope.)

**Second-guessing that number.** All 4 phrasings for a given condition
share thematic vocabulary ("evaluated/test/benchmark/scored" vs.
"helpful/user/chatting"), so this is real generalization across wording but
not proof of generalization to a completely alien phrasing. More
concerning: I noticed the eval-aware phrasings are systematically longer
than the deployment ones — 8-19 tokens longer, on every template, for both
tokenizers checked (Qwen2.5 and Phi-3.5's, which differ). A probe reading
sequence length (via absolute position) rather than content could produce
exactly this kind of high, template-general accuracy without representing
anything about "evaluation." I tested this directly: a **length-only**
linear control, same leave-one-template-out CV, same models, gets **52.8%
(Qwen) and 52.5% (Phi-3.5)** — indistinguishable from the 50% chance floor
for this balanced binary task. Per-question variation (2-13 answer choices)
swamps the small, consistent length gap, so length alone can't explain the
99.48%. That doesn't prove the probe has found something conceptually deep,
but it rules out the most obvious cheap alternative explanation.

## What I'd do next with more time

- A stronger control: length-*matched* templates (pad/trim to equal token
  counts) rather than a post-hoc length regression, to close this off
  entirely rather than statistically.
- A genuinely out-of-distribution phrasing — non-English, or an implicit/
  indirect evaluation cue that shares no vocabulary with the training set —
  to test how far the "generalization" really extends.
- A causal check: does adding the probe's learned direction to a
  deployment-framed prompt shift behavior the way real evaluation-awareness
  (sandbagging) would predict? Right now this is a correlational read-out,
  not an intervention.
- More models in Case 2 and 3 to get real resolution on the model-clustering
  dimension — 3 models is enough to demonstrate the method but too few to
  trust that particular component of the interval.
- A bounded (e.g. logit-transformed) version of the combined CI to fix the
  >100% overshoot properly instead of footnoting it.

## Why this seems worth doing

The tool itself is generic — drop in any (model, prompt, outcome) table and
get four CIs instead of one. What's interesting is that the same
~1.8-4.5x understatement shows up across three unrelated kinds of claims:
a static benchmark score, a live behavioral effect, and a probe's accuracy
on internal activations. That consistency is weak evidence that this is a
structural property of how the field reports these numbers — not clustering
by model, not clustering by prompt, and specifically not both — rather than
an artifact of any one experiment's design. If that generalizes, it means a
non-trivial fraction of "here's a clean effect" claims in interpretability
and evals are less certain than they look, in a way that's cheap to check
and currently mostly isn't.
