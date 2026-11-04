"""
Supervised fake-review classifier.

Pivot from the one-class anomaly-detection approach (which gave inverted
AUROC < 0.5 across autoencoder / One-Class SVM / k-NN cosine distance).
Trains directly on OR (genuine) vs CG (fake) labels using cached MiniLM
embeddings, compares Logistic Regression vs a small MLP.
"""

import os

import joblib  # saves/loads trained scikit-learn models to/from disk
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier   # a small feed-forward neural net classifier
from sklearn.pipeline import Pipeline               # chains preprocessing + model into one object
from sklearn.preprocessing import StandardScaler    # rescales features to mean 0 / std 1
import matplotlib.pyplot as plt

SEED = 42
EMBED_CACHE_PATH = "results/embeddings_cache.npz"
CONFUSION_MATRIX_PATH = "results/confusion_matrices.png"
MLP_MODEL_PATH = "results/mlp_classifier.joblib"  # where the trained MLP gets saved, so later
                                                   # scripts can reuse it instead of retraining
TEST_SIZE = 0.2


def load_dataset():
    """
    Build the labeled dataset (X = embeddings, y = 0/1 labels) from the
    cached embeddings produced by one_class_ae_gono_go.py. Unlike that
    script, here we use ALL genuine embeddings (both its "train" and "test"
    splits) as training data, since this script draws its own separate
    train/test split below.
    """
    cached = np.load(EMBED_CACHE_PATH)
    or_embeddings = np.vstack([cached["or_train"], cached["or_test"]])  # stack rows into one array
    cg_embeddings = cached["cg"]

    X = np.vstack([or_embeddings, cg_embeddings])
    y = np.concatenate(
        [np.zeros(len(or_embeddings)), np.ones(len(cg_embeddings))]
    )  # OR=0 genuine, CG=1 fake
    return X, y


def get_mlp(X_train, y_train, model_path=MLP_MODEL_PATH):
    """
    Return a trained MLP pipeline, reusing a cached one from disk if it
    exists so repeat runs (and other scripts) don't have to retrain it.
    `model_path` defaults to the shared baseline path but can be overridden
    so other scripts can cache differently-trained MLPs under their own
    filenames without overwriting the baseline.
    """
    if os.path.exists(model_path):
        print(f"Loading cached MLP model from {model_path}")
        return joblib.load(model_path)

    print(f"No cached model at {model_path} — training MLP (128) from scratch")
    # A Pipeline bundles preprocessing and the classifier into a single object,
    # so calling .fit()/.predict() on it always applies both steps in order,
    # and you can't accidentally forget to scale new data before predicting.
    model = Pipeline(
        [
            ("scaler", StandardScaler()),  # standardize embedding dimensions before training
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(128,),  # one hidden layer of 128 neurons
                    random_state=SEED,
                    max_iter=300,               # cap on training iterations
                    early_stopping=True,        # stop early if validation score plateaus
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)  # persist the fitted model (weights + scaler) to disk
    print(f"Trained and saved MLP model to {model_path}")
    return model


def evaluate(name, model, X_test, y_test):
    """Score a fitted model on held-out data and return the standard classification metrics."""
    y_pred = model.predict(X_test)                  # hard 0/1 predictions
    y_proba = model.predict_proba(X_test)[:, 1]      # predicted probability of being "fake" (class 1)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=1),  # of predicted fakes, how many were actually fake
        "recall": recall_score(y_test, y_pred, pos_label=1),        # of actual fakes, how many were caught
        "f1": f1_score(y_test, y_pred, pos_label=1),                # harmonic mean of precision and recall
        "auroc": roc_auc_score(y_test, y_proba),                    # ranking-quality score, uses probabilities not hard labels
    }
    cm = confusion_matrix(y_test, y_pred)  # 2x2 table: actual vs predicted counts
    return metrics, cm


def main():
    print(f"Loading cached embeddings from {EMBED_CACHE_PATH}")
    X, y = load_dataset()
    print(f"Total: {X.shape[0]} samples, {X.shape[1]}-dim embeddings "
          f"({int((y == 0).sum())} genuine, {int((y == 1).sum())} fake)")

    # stratify=y keeps the genuine/fake ratio the same in both the train and
    # test splits (important since we want a representative, fair test set).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]} (stratified, seed={SEED})")

    log_reg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )
    print("\nTraining Logistic Regression...")
    log_reg.fit(X_train, y_train)

    print("\nMLP (128)...")
    mlp = get_mlp(X_train, y_train)  # loads from disk if already trained, else trains + saves

    # A dict keyed by model name lets the loop below evaluate and report on
    # both models identically without repeating code.
    models = {"Logistic Regression": log_reg, "MLP (128)": mlp}

    results = {}
    confusion_matrices = {}
    for name, model in models.items():
        metrics, cm = evaluate(name, model, X_test, y_test)
        results[name] = metrics
        confusion_matrices[name] = cm

    print("\n=== RESULTS (test set, positive class = fake/CG) ===")
    # Build a formatted results table: column widths via f-string alignment
    # specs (<22 = left-align in 22 chars, >12.4f = right-align, 4 decimals).
    header = f"{'Model':<22}" + "".join(f"{m:>12}" for m in
                                         ["accuracy", "precision", "recall", "f1", "auroc"])
    print(header)
    for name, metrics in results.items():
        row = f"{name:<22}" + "".join(f"{metrics[m]:>12.4f}" for m in
                                       ["accuracy", "precision", "recall", "f1", "auroc"])
        print(row)

    # Draw one confusion-matrix panel per model, side by side, and save as one image.
    fig, axes = plt.subplots(1, len(confusion_matrices), figsize=(6 * len(confusion_matrices), 5))
    for ax, (name, cm) in zip(axes, confusion_matrices.items()):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["genuine (OR)", "fake (CG)"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    print(f"\nSaved confusion matrices to {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
