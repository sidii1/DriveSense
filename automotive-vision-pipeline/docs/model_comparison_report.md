# Model Comparison: Model A vs Model B vs Model C (SYNTHETIC development artifact)

> **WARNING.** The labels used here come from `generate_multimodal_dataset()` in `src/fusion_model.py`, a synthetic
> generator whose label is a formula of the input features. The numbers below therefore measure how well a model
> recovers that formula, **not** crash risk on real roads. They must not be quoted as evidence that fusing vision
> with a crash-history prior helps. The real-data evaluation is in `experiments/exp3_ccd` (Car Crash Dataset,
> early collision anticipation) and `experiments/exp1_crss` (NHTSA CRSS, injury severity).

## 1. Synthetic benchmark (development only)

| Model | Input | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Model A | Tabular crash-history features (synthetic) | 0.6000 | 0.0191 | 0.0370 | 0.7710 | 0.1768 |
| Model B | Vision features (synthetic) | 0.8614 | 0.5541 | 0.6744 | 0.9811 | 0.7901 |
| Model C | Tabular + vision (synthetic) | 0.8400 | 0.6688 | 0.7447 | 0.9878 | 0.8567 |

## 2. How to read this

Differences between rows reflect how much of the synthetic label formula each input set contains. No claim about
real-world crash prediction follows from them.
