"""
Leave-one-generator-out experiment (both directions).

unseen_generator_test.py showed the GPT-2-only-trained MLP catches 86.6% of
held-out GPT-2 fakes but only 19.4% of unseen Gemini fakes. This script asks:
if a SECOND modern generator's fakes are added to training, does detection of
a THIRD, still-unseen modern generator improve?

  Direction 1: train on OR + GPT-2 + Claude, test recall on unseen Gemini.
  Direction 2: train on OR + GPT-2 + Gemini, test recall on unseen Claude.

Both directions also report recall on held-out genuine (false-positive check)
and held-out GPT-2, and are compared against the original GPT-2-only baseline
model evaluated on the same eval sets.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

from supervised_classifier import (
    EMBED_CACHE_PATH,
    SEED,
    TEST_SIZE,
    get_mlp,
    load_dataset,
)

GEMINI_PATH = "data/fakes_gemini.csv"
CLAUDE_PATH = "data/fakes_claude.csv"
GEMINI_EMBED_CACHE_PATH = "results/gemini_fakes_embeddings.npz"
CLAUDE_EMBED_CACHE_PATH = "results/claude_fakes_embeddings.npz"
MLP_D1_PATH = "results/mlp_d1_or_gpt2_claude.joblib"  # trained with Claude, tests on Gemini
MLP_D2_PATH = "results/mlp_d2_or_gpt2_gemini.joblib"  # trained with Gemini, tests on Claude
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def embed_and_cache(csv_path: str, cache_path: str) -> np.ndarray:
    """
    Embed the `text` column of a fake-reviews CSV with MiniLM, caching the
    result to disk so repeat runs don't re-embed. Any other columns
    (category, stars, label) are ignored — every row here is fake by
    construction, regardless of what its label column says.
    """
    try:
        cached = np.load(cache_path)
        print(f"Loaded cached embeddings from {cache_path}")
        return cached["embeddings"]
    except FileNotFoundError:
        pass

    df = pd.read_csv(csv_path)
    texts = df["text"].tolist()
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"Embedding {len(texts)} reviews from {csv_path}...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    np.savez(cache_path, embeddings=embeddings)
    print(f"Cached embeddings to {cache_path}")
    return embeddings


def detection_rate(model, X: np.ndarray) -> float:
    """Fraction of rows (all known-fake) the model predicts as fake (1)."""
    return (model.predict(X) == 1).mean()


def genuine_accuracy(model, X: np.ndarray) -> float:
    """Fraction of rows (all known-genuine) the model predicts as genuine (0) — false-positive check."""
    return (model.predict(X) == 0).mean()


def main():
    print(f"Loading cached OR/CG embeddings from {EMBED_CACHE_PATH}")
    X, y = load_dataset()
    # Same split params used throughout the project, so OR_test/CG_test below
    # are the exact same held-out rows the documented baseline numbers came from.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    OR_train, CG_train = X_train[y_train == 0], X_train[y_train == 1]
    OR_test, CG_test = X_test[y_test == 0], X_test[y_test == 1]
    print(f"OR train/test: {len(OR_train)}/{len(OR_test)}  "
          f"CG train/test: {len(CG_train)}/{len(CG_test)}")

    gemini_emb = embed_and_cache(GEMINI_PATH, GEMINI_EMBED_CACHE_PATH)
    claude_emb = embed_and_cache(CLAUDE_PATH, CLAUDE_EMBED_CACHE_PATH)
    print(f"Gemini fakes: {len(gemini_emb)}  Claude fakes: {len(claude_emb)}")

    # --- Baseline: existing GPT-2-only model, evaluated on everything ---
    print("\n--- Baseline (GPT-2 only) ---")
    baseline = get_mlp(X_train, y_train)  # loads results/mlp_classifier.joblib
    baseline_or = genuine_accuracy(baseline, OR_test)
    baseline_cg = detection_rate(baseline, CG_test)
    baseline_gemini = detection_rate(baseline, gemini_emb)
    baseline_claude = detection_rate(baseline, claude_emb)

    # --- Direction 1: OR + GPT-2 + Claude -> test on unseen Gemini ---
    print("\n--- Direction 1: train OR+GPT-2+Claude, test on unseen Gemini ---")
    X_train_d1 = np.vstack([OR_train, CG_train, claude_emb])
    y_train_d1 = np.concatenate(
        [np.zeros(len(OR_train)), np.ones(len(CG_train)), np.ones(len(claude_emb))]
    )
    model_d1 = get_mlp(X_train_d1, y_train_d1, model_path=MLP_D1_PATH)
    d1_or = genuine_accuracy(model_d1, OR_test)
    d1_cg = detection_rate(model_d1, CG_test)
    d1_gemini = detection_rate(model_d1, gemini_emb)

    # --- Direction 2: OR + GPT-2 + Gemini -> test on unseen Claude ---
    print("\n--- Direction 2: train OR+GPT-2+Gemini, test on unseen Claude ---")
    X_train_d2 = np.vstack([OR_train, CG_train, gemini_emb])
    y_train_d2 = np.concatenate(
        [np.zeros(len(OR_train)), np.ones(len(CG_train)), np.ones(len(gemini_emb))]
    )
    model_d2 = get_mlp(X_train_d2, y_train_d2, model_path=MLP_D2_PATH)
    d2_or = genuine_accuracy(model_d2, OR_test)
    d2_cg = detection_rate(model_d2, CG_test)
    d2_claude = detection_rate(model_d2, claude_emb)

    print("\n=== LEAVE-ONE-GENERATOR-OUT COMPARISON ===")
    header = f"{'Model':<28}{'genuine held-out':>18}{'GPT-2 held-out':>18}{'Gemini det.':>14}{'Claude det.':>14}"
    print(header)
    print(f"{'Baseline (GPT-2 only)':<28}{baseline_or:>18.4f}{baseline_cg:>18.4f}"
          f"{baseline_gemini:>14.4f}{baseline_claude:>14.4f}")
    print(f"{'Direction 1 (+Claude)':<28}{d1_or:>18.4f}{d1_cg:>18.4f}"
          f"{d1_gemini:>14.4f}{'n/a':>14}")
    print(f"{'Direction 2 (+Gemini)':<28}{d2_or:>18.4f}{d2_cg:>18.4f}"
          f"{'n/a':>14}{d2_claude:>14.4f}")

    print("\n--- Does adding one modern generator help detect a different unseen one? ---")
    print(f"Gemini detection: baseline {baseline_gemini:.4f} -> Direction 1 (+Claude) {d1_gemini:.4f} "
          f"(change {d1_gemini - baseline_gemini:+.4f})")
    print(f"Claude detection: baseline {baseline_claude:.4f} -> Direction 2 (+Gemini) {d2_claude:.4f} "
          f"(change {d2_claude - baseline_claude:+.4f})")


if __name__ == "__main__":
    main()
