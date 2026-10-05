# Predictive Maintenance Classifier: UCI AI4I 2020

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange)
![Model](https://img.shields.io/badge/Model-Random%20Forest-green)
![Validation](https://img.shields.io/badge/Validation-5--fold%20stratified%20CV-lightgrey)

A machine-failure classifier for a milling machine, built on the UCI AI4I 2020 predictive-maintenance dataset. Failures are only **3.4%** of the data, so the project focuses on imbalance-aware training, leakage prevention, honest evaluation, and **physics-based feature engineering** that lifted cross-validated recall from **51% to 80%**.

> **TL;DR:** a Random Forest with class weighting and three process-physics features reaches **98% precision, 80% recall, F1 0.88 and ROC-AUC 0.97** (5-fold stratified CV), against a raw-feature baseline of 92% precision, 51% recall and F1 0.65.

---

## Why this problem matters

An unplanned breakdown is far more expensive than an unnecessary inspection, so the model must catch real failures (**recall**) without raising so many false alarms (**precision**) that the maintenance team stops trusting it. Accuracy is the wrong metric here: predicting "no failure" for every cycle is 96.6% accurate and catches nothing.

## Dataset

| Property | Value |
|---|---|
| Source | UCI AI4I 2020 Predictive Maintenance Dataset (also on [Kaggle](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)) |
| Rows | 10,000 production cycles (synthetic, modelled on a real milling machine) |
| Raw features | Air temperature [K], process temperature [K], rotational speed [rpm], torque [Nm], tool wear [min], product type (L/M/H) |
| Target | `Machine failure` (1 = failure), 3.39% positive (339 cases) |
| Not used as inputs | `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (see leakage note below) |

The CSV is not stored in this repo. Download `ai4i2020.csv` from the link above and place it in the project root.

## Approach

1. **Clean and encode:** drop `UDI` and `Product ID` (identifiers), encode `Type` as L=0, M=1, H=2.
2. **Prevent leakage:** the five failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) tell you *which* failure occurred, which is only known after the fact. They are excluded, so the model cannot cheat.
3. **Stratified 80/20 split** (`stratify=y`, `random_state=42`) keeps the 3.4% failure ratio in train and test.
4. **Handle imbalance** with `RandomForestClassifier(n_estimators=200, class_weight="balanced")`.
5. **Baseline comparison** against a class-weighted Logistic Regression.
6. **Feature engineering** from the machine's physics (below).
7. **Evaluate** with precision, recall, F1, ROC-AUC and a confusion matrix, then confirm with **5-fold stratified cross-validation**.

### Engineered features

| Feature | Formula | Targets failure mode |
|---|---|---|
| Power | torque × rotational speed × 2π/60 | Power failure (PWF) |
| Temperature difference | process temperature − air temperature | Heat dissipation failure (HDF) |
| Wear × torque | tool wear × torque | Overstrain failure (OSF) |

These encode interactions a tree ensemble would otherwise have to learn from only 339 positive examples.

## Results

### Baseline (raw 6 features, single 80/20 split)

| Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|
| 0.929 | 0.574 | 0.709 | 0.969 |

Confusion matrix (test set, 68 failures): `[[1929, 3], [29, 39]]`, so 3 false alarms and 29 missed failures. Logistic Regression on the same features reached an F1 of only **0.24**, because a linear model cannot capture the torque × speed interaction.

### Final comparison (5-fold stratified CV, threshold 0.5)

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest, raw 6 features | 0.916 ± 0.044 | 0.510 ± 0.055 | 0.652 ± 0.038 | 0.964 ± 0.010 |
| **Random Forest + engineered features** | **0.979 ± 0.020** | **0.797 ± 0.031** | **0.878 ± 0.012** | **0.973 ± 0.002** |

Recall improved by about 29 points and F1 by about 0.23, far more than the fold-to-fold spread, and precision rose as well.

### Feature importance (baseline model)

Torque 0.338 · Rotational speed 0.277 · Tool wear 0.209 · Air temperature 0.099 · Process temperature 0.063 · Type 0.014

Torque, speed and tool wear dominate, which is what motivated the interaction features above.

### Threshold tuning

Lowering the decision threshold trades precision for recall. On the engineered model (single split) moving from 0.5 to 0.3 changed F1 only from 0.852 to 0.855, so the default 0.5 is reported. The threshold was explored on the test split, so treat that figure as indicative.

## Key takeaways

- With 3.4% positives, **accuracy hides failure**: use precision, recall, F1 and PR/ROC-AUC.
- **Excluding label-leaking columns** is essential for a valid model.
- **Domain knowledge beat model complexity:** three physics features did more than any tuning.
- **Cross-validation matters:** the single baseline split (recall 0.57) was optimistic compared with the CV estimate (0.51).

## Limitations

- The data is **synthetic**; results need validation on real plant data before any deployment claim.
- The dataset includes **random failures (`RNF`)** that no sensor-based model can predict, so recall has a ceiling below 100%.
- Only 339 failures, so metrics are sensitive to the split (hence the cross-validation).

## Repository structure

```
.
├── README.md
├── requirements.txt
├── predictive_maintenance.ipynb   # full analysis notebook (EDA → model → CV)
├── docs/
│   └── AI4I_Predictive_Maintenance.pptx   # project summary slides
└── ai4i2020.csv                   # download separately (not committed)
```

## How to run

### Option A: Google Colab (no setup)
1. Open the notebook in Colab.
2. Run the first cell and upload `ai4i2020.csv` when prompted.
3. Run all cells.

### Option B: local
```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
jupyter notebook
```
Put `ai4i2020.csv` next to the notebook, then run all cells.

### requirements.txt
```
scikit-learn==1.6.1
pandas==2.2.3
numpy==2.1.3
matplotlib
seaborn
jupyter
imbalanced-learn   # only for the optional SMOTE experiment
```

Results use `random_state=42`. With the pinned versions above they should match the tables here; other library versions may change the last decimals.

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter / Google Colab

## Possible next steps

- Compare class weighting with SMOTE oversampling
- Try gradient boosting (XGBoost / LightGBM)
- Explain predictions with SHAP and plot importance for the engineered model
- Choose the threshold from maintenance cost (cost of a miss vs a false alarm) using a validation set

## Dataset citation

S. Matzka, *AI4I 2020 Predictive Maintenance Dataset*, UCI Machine Learning Repository. Please verify the exact citation (title, venue and year) on the dataset page before publishing.

## Author

**Harsh Paswan**, B.Tech Chemical Engineering, BIT Mesra
Process-engineering interests: ammonia and chlor-alkali plants, with data analytics in Python, SQL and Power BI.
