"""
Go/no-go test: one-class autoencoder for fake-review detection.

Trains an autoencoder ONLY on genuine (OR) review embeddings, then checks
whether fake (CG) reviews reconstruct worse than held-out genuine reviews.
Reports AUROC and saves a histogram of the two reconstruction-error
distributions. This is a feasibility check, not a tuned model.
"""

import random

import matplotlib.pyplot as plt          # for drawing the results histogram
import numpy as np                       # array math (embeddings are numpy arrays)
import pandas as pd                      # reads the CSV of reviews into a table (DataFrame)
import torch                             # the neural-network library (PyTorch)
import torch.nn as nn                    # building blocks for defining a neural network
from sentence_transformers import SentenceTransformer  # turns review text into numeric vectors
from sklearn.metrics import roc_auc_score               # scores how well two groups are separated
from sklearn.model_selection import train_test_split    # splits a dataset into train/test chunks

# --- Fixed settings, kept together at the top so they're easy to find/tweak ---
SEED = 42                                          # fixes randomness so results are reproducible
DATA_PATH = "data/fake_reviews.csv"                # source reviews (genuine + GPT-2 fakes)
EMBED_CACHE_PATH = "results/embeddings_cache.npz"  # where computed embeddings get saved/reused
HIST_PATH = "results/reconstruction_error_histogram.png"  # where the output chart is saved
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"              # the pretrained model that turns text into vectors
EPOCHS = 50                                        # how many full passes over the training data
BATCH_SIZE = 128                                   # how many reviews are processed at once during training
LATENT_DIM = 32                                    # size of the "compressed" representation in the middle
HIDDEN_DIM = 128                                   # size of the network's hidden layers


def set_seed(seed: int) -> None:
    """Make every source of randomness (Python, numpy, PyTorch) deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Autoencoder(nn.Module):
    """
    A small neural network that tries to compress an embedding down to
    LATENT_DIM numbers and then rebuild the original embedding from that
    compressed version. Trained only on genuine reviews, so it should get
    good at rebuilding genuine-review patterns specifically. The idea being
    tested: fake reviews might rebuild worse because they don't fit the
    pattern the network learned.
    """

    # nn.Module is PyTorch's base class for "a thing with learnable weights".
    # Subclassing it and calling super().__init__() wires up all the internal
    # bookkeeping PyTorch needs (tracking parameters, enabling .train()/.eval(), etc.)
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        # encoder: shrinks the embedding step by step (input -> hidden -> latent)
        # nn.Sequential just chains layers so data flows through them in order.
        # nn.Linear is a standard fully-connected layer; nn.ReLU is a nonlinearity
        # that lets the network learn more than plain straight-line transformations.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        # decoder: mirrors the encoder in reverse (latent -> hidden -> back to input size)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    # forward() is the method PyTorch calls when you do model(some_input).
    # It defines what happens to data as it passes through the network.
    def forward(self, x):
        return self.decoder(self.encoder(x))


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Convert a list of review strings into a 2D array of numeric vectors (one row per review)."""
    return model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def get_embeddings(or_train_texts, or_test_texts, cg_texts):
    """
    Embedding the whole dataset is slow, so this checks disk first (a cache file)
    and only runs the (slow) embedding model if no cache exists yet.
    """
    try:
        cached = np.load(EMBED_CACHE_PATH)
        print(f"Loaded cached embeddings from {EMBED_CACHE_PATH}")
        return cached["or_train"], cached["or_test"], cached["cg"]
    except FileNotFoundError:
        # No cache yet — fall through and compute embeddings from scratch below.
        pass

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print(f"Embedding {len(or_train_texts)} OR-train reviews...")
    or_train_emb = embed_texts(model, or_train_texts)
    print(f"Embedding {len(or_test_texts)} OR-test reviews...")
    or_test_emb = embed_texts(model, or_test_texts)
    print(f"Embedding {len(cg_texts)} CG (fake) reviews...")
    cg_emb = embed_texts(model, cg_texts)

    # Save all three arrays into one .npz file (numpy's "zip of arrays" format)
    # so future runs of any script can load them instantly instead of re-embedding.
    np.savez(
        EMBED_CACHE_PATH,
        or_train=or_train_emb,
        or_test=or_test_emb,
        cg=cg_emb,
    )
    print(f"Cached embeddings to {EMBED_CACHE_PATH}")
    return or_train_emb, or_test_emb, cg_emb


def train_autoencoder(train_embeddings: np.ndarray) -> Autoencoder:
    """Train the autoencoder on genuine-review embeddings only, for EPOCHS passes."""
    device = torch.device("cpu")
    input_dim = train_embeddings.shape[1]
    model = Autoencoder(input_dim, HIDDEN_DIM, LATENT_DIM).to(device)
    # Adam is the optimization algorithm that nudges the network's weights
    # to reduce the loss after each batch. lr = "learning rate" = step size.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # MSELoss = mean squared error: measures how far the rebuilt embedding is
    # from the original — this is exactly what we want the network to minimize.
    loss_fn = nn.MSELoss()

    # Convert the numpy array into a PyTorch tensor (PyTorch's array type,
    # required for anything that goes through the network).
    x_train = torch.tensor(train_embeddings, dtype=torch.float32, device=device)
    n = x_train.shape[0]

    model.train()  # puts the model in "training mode" (matters for some layer types)
    for epoch in range(1, EPOCHS + 1):
        # Shuffle the data order every epoch so the network doesn't learn
        # anything from the fixed ordering of examples.
        perm = torch.randperm(n)
        epoch_loss = 0.0
        # Walk through the shuffled data BATCH_SIZE examples at a time.
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            batch = x_train[idx]

            optimizer.zero_grad()      # clear leftover gradients from the previous batch
            recon = model(batch)       # run the batch through encoder+decoder
            loss = loss_fn(recon, batch)  # how different is the rebuild from the original?
            loss.backward()            # compute how each weight contributed to the error
            optimizer.step()           # adjust the weights to reduce that error

            epoch_loss += loss.item() * batch.shape[0]

        # Print progress occasionally rather than every single epoch.
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"  epoch {epoch:3d}/{EPOCHS}  train MSE = {epoch_loss / n:.5f}")

    return model


def reconstruction_errors(model: Autoencoder, embeddings: np.ndarray) -> np.ndarray:
    """For each embedding, how badly did the trained model rebuild it? (one score per row)"""
    model.eval()  # "evaluation mode" — turns off training-only behavior
    x = torch.tensor(embeddings, dtype=torch.float32)
    # no_grad(): we're not training here, so skip the (expensive) bookkeeping
    # PyTorch normally does to support backward()/gradients.
    with torch.no_grad():
        recon = model(x)
        errors = torch.mean((recon - x) ** 2, dim=1).numpy()  # MSE per row, back to numpy
    return errors


def main():
    set_seed(SEED)

    print(f"Loading {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Split the full table into genuine (OR) and fake (CG) rows by their label column.
    or_df = df[df["label"] == "OR"]
    cg_df = df[df["label"] == "CG"]

    # The autoencoder only ever trains on genuine reviews. We still hold out
    # 20% of those as "OR test" so we can compare, at evaluation time, whether
    # UNSEEN genuine reviews reconstruct better than fake ones — a fair test,
    # since the model hasn't memorized those either.
    or_train_df, or_test_df = train_test_split(
        or_df, test_size=0.2, random_state=SEED
    )
    print(
        f"OR train: {len(or_train_df)}  OR test: {len(or_test_df)}  "
        f"CG (fake, eval-only): {len(cg_df)}"
    )

    # tolist() converts a pandas column into a plain Python list of strings,
    # which is what the embedding model expects.
    or_train_emb, or_test_emb, cg_emb = get_embeddings(
        or_train_df["text_"].tolist(),
        or_test_df["text_"].tolist(),
        cg_df["text_"].tolist(),
    )

    print(f"Training autoencoder on {or_train_emb.shape[0]} genuine embeddings "
          f"(dim={or_train_emb.shape[1]})")
    model = train_autoencoder(or_train_emb)

    or_test_errors = reconstruction_errors(model, or_test_emb)
    cg_errors = reconstruction_errors(model, cg_emb)

    # Build one combined array of true labels (0=genuine, 1=fake) and one of
    # scores (reconstruction error), so roc_auc_score can compare them directly.
    labels = np.concatenate(
        [np.zeros(len(or_test_errors)), np.ones(len(cg_errors))]
    )
    scores = np.concatenate([or_test_errors, cg_errors])
    # AUROC: probability that a random fake review scores higher (more
    # anomalous) than a random genuine review. 0.5 = no better than a coin
    # flip; 1.0 = perfect separation.
    auroc = roc_auc_score(labels, scores)

    print("\n=== RESULTS ===")
    print(f"OR-test  n={len(or_test_errors)}  "
          f"mean={or_test_errors.mean():.5f}  median={np.median(or_test_errors):.5f}")
    print(f"CG       n={len(cg_errors)}  "
          f"mean={cg_errors.mean():.5f}  median={np.median(cg_errors):.5f}")
    print(f"AUROC (genuine=0 vs fake=1, score=reconstruction error): {auroc:.4f}")

    # Overlaid histograms: if the approach worked, the two colored distributions
    # would sit in different places along the x-axis (fake = higher error).
    plt.figure(figsize=(8, 5))
    plt.hist(or_test_errors, bins=50, alpha=0.6, label="OR (genuine, held-out)", density=True)
    plt.hist(cg_errors, bins=50, alpha=0.6, label="CG (fake)", density=True)
    plt.xlabel("Reconstruction error (MSE)")
    plt.ylabel("Density")
    plt.title(f"Reconstruction error: genuine vs fake reviews (AUROC={auroc:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(HIST_PATH, dpi=150)
    print(f"Saved histogram to {HIST_PATH}")


# This guard means "only run main() if this file is executed directly"
# (e.g. `python one_class_ae_gono_go.py`), not if another file imports from it.
if __name__ == "__main__":
    main()
