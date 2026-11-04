# Fake Review Detector — Results Summary

**Data**: `data/fake_reviews.csv` — 40,432 reviews, balanced (20,216 genuine `OR`, 20,216 computer-generated `CG`).
**Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim), cached at `results/embeddings_cache.npz`.
**Seed**: 42 throughout (splits, model init).

## 1. One-class experiments (train on genuine `OR` only)

Hypothesis: model "normal" genuine reviews, flag fakes as anomalies. Evaluated on held-out `OR` (20%, n=4,044) vs all `CG` (n=20,216).

| Method | Script | AUROC | Notes |
|---|---|---:|---|
| Autoencoder reconstruction error | `src/one_class_ae_gono_go.py` | 0.378 | 384→128→32→128→384 MLP, MSE loss, 50 epochs |
| One-Class SVM (RBF, nu=0.05) | `src/one_class_score_comparison.py` | 0.423 | On L2-normalized embeddings |
| k-NN cosine distance (k=5) | `src/one_class_score_comparison.py` | 0.394 | Mean distance to 5 nearest genuine train neighbors |
| k-NN cosine distance (k=20) | `src/one_class_score_comparison.py` | 0.390 | Mean distance to 20 nearest genuine train neighbors |

Autoencoder error stats: OR-test mean=0.00127, median=0.00127 (n=4,044); CG mean=0.00114, median=0.00111 (n=20,216).

**Verdict: no-go.** All four scores are below 0.5 and in the same (inverted) direction — fake reviews are consistently *more* prototypical / closer to the genuine training distribution than held-out genuine reviews are, across every anomaly-scoring method tried. One-class / anomaly framing does not work for this dataset in MiniLM embedding space.

Artifact: `results/reconstruction_error_histogram.png`

## 2. Supervised classification (train on `OR` vs `CG` labels)

Pivoted to direct supervised learning on the same embeddings. Stratified 80/20 train/test split (n=32,345 train / n=8,087 test), positive class = fake (`CG`).

| Model | Accuracy | Precision | Recall | F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8343 | 0.8307 | 0.8397 | 0.8352 | 0.9128 |
| MLP (128 hidden units) | 0.8554 | 0.8479 | 0.8662 | 0.8570 | 0.9340 |

**Verdict: go.** Both models separate genuine from fake reviews strongly (AUROC 0.91–0.93), a large improvement over the one-class approach. The MLP outperforms logistic regression on every metric, though the near-linear separability (strong LogReg baseline) suggests the classes are cleanly separated in embedding space.

Artifact: `results/confusion_matrices.png`

## 3. Generalization to an unseen generator (Gemini)

The MLP above was trained only on genuine (`OR`) vs GPT-2-generated fake (`CG`) reviews. To
check whether it learned "fake-ness" in general or just GPT-2's specific fingerprint, it was
run — with no retraining or fine-tuning — on `data/modern_fakes.csv` (n=175), fake reviews
written by Gemini, a generator it never saw during training. Same MiniLM (`all-MiniLM-L6-v2`)
embeddings, same trained MLP, seed 42 throughout. Script: `src/unseen_generator_test.py`.

| Fakes | Generator seen in training? | n | Fraction flagged fake |
|---|---|---:|---:|
| `CG` (held-out test split) | Yes (GPT-2) | 4,043 | 0.8662 |
| `modern_fakes.csv` | No (Gemini) | 175 | 0.1943 |

**Gap: 0.6719.** The detector catches 87% of GPT-2 fakes but only 19% of Gemini fakes — it
almost entirely fails to generalize to the unseen generator. This indicates the MLP mostly
learned GPT-2's specific stylistic fingerprint (the artifacts of that particular model's
generation quirks circa the original dataset) rather than a generator-agnostic notion of what
makes a review "fake." Modern LLM-written fakes (Gemini) read close enough to genuine reviews
in MiniLM embedding space that this classifier, tuned on an older generator, mostly misses
them. A production detector would need either (a) training data spanning multiple/modern
generators, or (b) a different signal than embedding-space separability from one generator's
outputs.

## 4. Leave-one-generator-out: does a second modern generator transfer?

Section 3 showed the GPT-2-only model catches 19.4% of unseen Gemini fakes. This asks whether
adding fakes from *one* modern generator to training helps detection of a *different*,
still-unseen modern generator. Two directions, using `data/fakes_gemini.csv` (n=175) and
`data/fakes_claude.csv` (n=180) — `text` column only, every row treated as fake regardless of
its `label` column. Same MLP architecture, same MiniLM embeddings, seed 42 throughout. OR/CG
train and held-out test sets are the identical split used in sections 2–3. Script:
`src/leave_one_generator_out.py`.

| Model | Genuine held-out | GPT-2 held-out | Gemini detection | Claude detection |
|---|---:|---:|---:|---:|
| Baseline (GPT-2 only) | 0.8447 | 0.8662 | 0.1943 | 0.0722 |
| Direction 1: OR+GPT-2+Claude | 0.8524 | 0.8568 | **0.3086** | n/a (trained on all of it) |
| Direction 2: OR+GPT-2+Gemini | 0.8378 | 0.8746 | n/a (trained on all of it) | **0.1278** |

**Adding a second modern generator helps, but only partially.** Gemini detection roughly
doubles (0.1943 → 0.3086, +11.4pp) when Claude fakes join training; Claude detection also
roughly doubles (0.0722 → 0.1278, +5.6pp) when Gemini fakes join training. Genuine
held-out accuracy and GPT-2 held-out recall both stay in the same ~0.84–0.87 band across all
three models — the added generator doesn't damage performance on what the model already knew.
But even with a second modern generator in training, both directions still miss the large
majority (69–87%) of the held-out third generator's fakes. This suggests real but limited
transfer: some of what the MLP learns from one modern LLM's fakes generalizes to another, but
each new generator still carries enough of its own stylistic fingerprint that seeing it
directly in training matters far more than any cross-generator transfer observed here.

## Overall conclusion

One-class/anomaly detection is not a viable approach for this dataset — fake reviews are embedding-space *insiders*, not outliers, relative to genuine reviews. Standard supervised classification on MiniLM embeddings works well, but section 4 shows it doesn't generalize cleanly to unseen LLM generators — a production detector needs training data spanning many/current generators, not just one or two, and should expect meaningful misses against future generators not yet represented in training.
