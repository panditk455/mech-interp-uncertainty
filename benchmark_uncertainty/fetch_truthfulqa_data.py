"""Build a real model x prompt eval matrix from the public Open LLM Leaderboard
(v1) per-item detail logs on HuggingFace — free, no inference needed.

Each `open-llm-leaderboard-old/details_<model>` dataset repo contains, per model,
a parquet of per-question results on TruthfulQA-MC1: the same fixed 817 questions
for every model, each with a `mc1` boolean (True = model selected the correct
answer). We treat `outcome = 1 - mc1` (1 = hallucinated / picked a wrong answer)
as the binary outcome, `question` as the prompt-cluster id, and the model name
as the model-cluster id — exactly the schema clustered_bootstrap_evals.py expects.

Usage:
    python -m benchmark_uncertainty.fetch_truthfulqa_data --out data/truthfulqa_mc1.csv
"""

import argparse

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "tiiuae/falcon-7b",
    "tiiuae/falcon-40b",
    "google/gemma-7b",
    "microsoft/phi-2",
    "01-ai/Yi-6B",
    "EleutherAI/gpt-j-6b",
]

TASK_MARKER = "details_harness|truthfulqa:mc|0"


def _truthfulqa_files(api, repo_id):
    info = api.dataset_info(repo_id)
    candidates = [s.rfilename for s in info.siblings if TASK_MARKER in s.rfilename]
    if not candidates:
        raise FileNotFoundError(f"no truthfulqa:mc file in {repo_id}")
    return candidates


def fetch_model(api, model_name):
    """Different harness versions used different column names/shapes for this
    task (question vs. example; a top-level mc1 bool vs. a nested metrics.mc1).
    Some runs are also quick smoke-test subsets (as few as 10 rows) rather than
    the full 817-question set — download every candidate file and keep the
    largest, since that's the real run.
    """
    repo_id = "open-llm-leaderboard-old/details_" + model_name.replace("/", "__")
    best_df = None
    for filename in _truthfulqa_files(api, repo_id):
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        df = pd.read_parquet(path)
        if best_df is None or len(df) > len(best_df):
            best_df = df

    question = best_df["question"] if "question" in best_df.columns else best_df["example"]
    if "mc1" in best_df.columns:
        mc1 = best_df["mc1"].astype(bool)
    else:
        mc1 = best_df["metrics"].apply(lambda m: bool(m["mc1"]))

    return pd.DataFrame({"model": model_name, "question": question, "outcome": (~mc1).astype(int)})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/truthfulqa_mc1.csv")
    args = parser.parse_args()

    api = HfApi()
    frames = []
    question_sets = {}
    for model_name in MODELS:
        try:
            df = fetch_model(api, model_name)
        except Exception as e:
            print(f"skipping {model_name}: {e}")
            continue
        frames.append(df)
        question_sets[model_name] = set(df["question"])
        print(f"{model_name}: {len(df)} questions, hallucination_rate={df['outcome'].mean():.3f}")

    common = set.intersection(*question_sets.values())
    dropped = sum(len(qs) - len(common) for qs in question_sets.values())
    if dropped:
        print(f"dropping {dropped} question rows not common to all {len(question_sets)} models")

    full = pd.concat(frames, ignore_index=True)
    full = full[full["question"].isin(common)]
    full["prompt"] = full["question"].astype("category").cat.codes.map(lambda i: f"q{i:04d}")
    full = full[["model", "prompt", "question", "outcome"]]

    full.to_csv(args.out, index=False)
    print(f"\nwrote {len(full)} rows ({full['model'].nunique()} models x {full['prompt'].nunique()} prompts) to {args.out}")


if __name__ == "__main__":
    main()
