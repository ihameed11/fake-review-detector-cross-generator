"""
Unseen-generator generalization test.

The supervised MLP detector (src/supervised_classifier.py) was trained only on
OR (genuine) vs CG (GPT-2-generated fake) reviews. This script checks whether
that detector generalizes to fakes from a generator it never saw during
training: data/modern_fakes.csv, reviews written by Gemini, a modern LLM.

Compares:
  - recall on held-out GPT-2 (CG) fakes (the generator it was trained on)
  - detection rate on Gemini fakes (an unseen generator)
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

# Reusing pieces straight from supervised_classifier.py instead of copying
# them means this script always stays in sync with however that file trains
# the model and loads its data.
from supervised_classifier import (
    EMBED_CACHE_PATH,
    SEED,
    TEST_SIZE,
    get_mlp,
    load_dataset,
)

MODERN_FAKES_PATH = "data/modern_fakes.csv"
MODERN_EMBED_CACHE_PATH = "results/modern_fakes_embeddings.npz"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def embed_modern_fakes(texts: list[str]) -> np.ndarray:
    """Embed the Gemini reviews with the same model used everywhere else, caching to disk."""
    try:
        cached = np.load(MODERN_EMBED_CACHE_PATH)
        print(f"Loaded cached modern-fakes embeddings from {MODERN_EMBED_CACHE_PATH}")
        return cached["embeddings"]
    except FileNotFoundError:
        # No cache yet — compute embeddings below instead.
        pass

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"Embedding {len(texts)} modern (Gemini) fake reviews...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    np.savez(MODERN_EMBED_CACHE_PATH, embeddings=embeddings)
    print(f"Cached embeddings to {MODERN_EMBED_CACHE_PATH}")
    return embeddings


def main():
    print(f"Loading cached OR/CG embeddings from {EMBED_CACHE_PATH}")
    X, y = load_dataset()
    # Same split params (test_size, stratify, random_state) as
    # supervised_classifier.py's main(), so X_test/y_test here are the exact
    # same held-out rows that script already reported metrics on — that's
    # what makes the GPT-2 recall number below directly comparable.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]} (stratified, seed={SEED})")

    model = get_mlp(X_train, y_train)  # loads the saved model if present, else trains + saves it

    # Recall on the generator the model WAS trained on (GPT-2 / CG).
    # Boolean masking: y_test == 1 makes an array of True/False, and indexing
    # with it keeps only the rows where that's True (i.e. only the fake ones).
    cg_mask = y_test == 1
    X_cg_test, y_cg_test = X_test[cg_mask], y_test[cg_mask]
    cg_pred = model.predict(X_cg_test)
    # Every true label here is 1 (fake), so "recall" simplifies to: what
    # fraction did the model also predict as 1?
    cg_recall = (cg_pred == 1).mean()
    print(f"\nHeld-out GPT-2 (CG) fakes: n={len(y_cg_test)}  recall={cg_recall:.4f}")

    # Detection rate on the unseen generator (Gemini / modern_fakes.csv).
    df = pd.read_csv(MODERN_FAKES_PATH)
    print(f"\nLoaded {len(df)} modern (Gemini) fake reviews from {MODERN_FAKES_PATH}")
    modern_embeddings = embed_modern_fakes(df["text"].tolist())
    modern_pred = model.predict(modern_embeddings)
    # Same idea: every row here is fake by construction, so this is just the
    # fraction the model flagged as fake — its "detection rate" on this file.
    modern_detection_rate = (modern_pred == 1).mean()
    print(f"Gemini (modern_fakes.csv): n={len(df)}  detection rate={modern_detection_rate:.4f}")

    gap = cg_recall - modern_detection_rate
    print("\n=== UNSEEN-GENERATOR COMPARISON ===")
    print(f"{'Generator':<20}{'n':>8}{'fraction flagged fake':>26}")
    print(f"{'GPT-2 (CG, trained on)':<20}{len(y_cg_test):>8}{cg_recall:>26.4f}")
    print(f"{'Gemini (unseen)':<20}{len(df):>8}{modern_detection_rate:>26.4f}")
    print(f"\nGap (GPT-2 recall - Gemini detection rate): {gap:+.4f}")


if __name__ == "__main__":
    main()
