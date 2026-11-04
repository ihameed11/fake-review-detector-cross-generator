"""
Go/no-go test: alternative one-class anomaly scores for fake-review detection.

Reuses the cached MiniLM embeddings (genuine train/test split, seed 42) from
one_class_ae_gono_go.py and compares two anomaly scores trained ONLY on
genuine (OR) embeddings:
  1. One-Class SVM (RBF kernel)
  2. Mean cosine distance to k nearest genuine training neighbors (k=5, k=20)

Reports AUROC for each against held-out genuine (OR-test) vs fake (CG).
"""

import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM

SEED = 42
EMBED_CACHE_PATH = "results/embeddings_cache.npz"
OCSVM_NU = 0.05          # One-Class SVM's expected fraction of "outliers" among genuine training data
KNN_KS = (5, 20)         # the two values of k to try for the k-nearest-neighbors score
QUERY_BATCH_SIZE = 2000  # how many reviews to score against the neighbor set at once (memory control)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """
    Rescale each embedding (each row) to length 1, so only its *direction*
    matters, not its magnitude. This makes dot products between rows equal to
    cosine similarity, which is what the neighbor-distance score below relies on.
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norms


def knn_cosine_distance_scores(train: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """
    For each row in `query`, find its k most similar rows in `train` (the
    genuine training embeddings) and turn that into an anomaly score: the
    further a review sits from its nearest genuine neighbors, the more
    "anomalous" it's considered.
    """
    n_query = query.shape[0]
    scores = np.empty(n_query, dtype=np.float64)
    # Process the query set in chunks (QUERY_BATCH_SIZE at a time) instead of
    # all at once, since the train-by-query similarity matrix could otherwise
    # get too large to hold in memory.
    for start in range(0, n_query, QUERY_BATCH_SIZE):
        end = min(start + QUERY_BATCH_SIZE, n_query)
        batch = query[start:end]
        # Since both sides are L2-normalized, a plain matrix multiply gives
        # cosine similarity between every train row and every batch row.
        sims = train @ batch.T  # (n_train, batch_size) cosine similarities (unit vectors)
        # np.partition is a fast partial sort: it only guarantees the top k
        # values end up in the last k positions, without fully sorting everything.
        top_k_sims = np.partition(sims, -k, axis=0)[-k:, :]  # (k, batch_size)
        # distance = 1 - similarity; average over the k neighbors.
        mean_dist = 1.0 - top_k_sims.mean(axis=0)
        scores[start:end] = mean_dist
    return scores


def main():
    np.random.seed(SEED)

    print(f"Loading cached embeddings from {EMBED_CACHE_PATH}")
    cached = np.load(EMBED_CACHE_PATH)
    or_train, or_test, cg = cached["or_train"], cached["or_test"], cached["cg"]
    print(f"OR train: {or_train.shape}  OR test: {or_test.shape}  CG: {cg.shape}")

    # Both scoring methods below rely on cosine geometry, so normalize
    # everything to unit length up front.
    or_train = l2_normalize(or_train)
    or_test = l2_normalize(or_test)
    cg = l2_normalize(cg)

    # Ground truth for evaluation: 0 = genuine (OR-test), 1 = fake (CG).
    # eval_embeddings lines up row-for-row with `labels`.
    labels = np.concatenate([np.zeros(len(or_test)), np.ones(len(cg))])
    eval_embeddings = np.vstack([or_test, cg])

    results = {}

    print(f"\nFitting One-Class SVM (RBF, nu={OCSVM_NU}) on {or_train.shape[0]} genuine embeddings...")
    t0 = time.time()
    # One-Class SVM learns a boundary around the genuine training embeddings.
    # nu controls how tightly it's allowed to fit (roughly, the expected
    # fraction of training points it tolerates as outliers).
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=OCSVM_NU)
    ocsvm.fit(or_train)
    print(f"  fit took {time.time() - t0:.1f}s")

    # decision_function: positive = inside the "normal" boundary, negative =
    # outside it. We flip the sign so higher score consistently means "more
    # anomalous," matching the convention used by the k-NN score below.
    decision = ocsvm.decision_function(eval_embeddings)
    ocsvm_scores = -decision  # higher = more anomalous
    results["One-Class SVM"] = roc_auc_score(labels, ocsvm_scores)

    for k in KNN_KS:
        print(f"\nComputing mean cosine distance to {k} nearest genuine neighbors...")
        t0 = time.time()
        knn_scores = knn_cosine_distance_scores(or_train, eval_embeddings, k)
        print(f"  scoring took {time.time() - t0:.1f}s")
        results[f"k-NN cosine distance (k={k})"] = roc_auc_score(labels, knn_scores)

    print("\n=== RESULTS ===")
    print(f"{'Method':<32} {'AUROC':>8}")
    for name, auroc in results.items():
        print(f"{name:<32} {auroc:>8.4f}")


if __name__ == "__main__":
    main()
