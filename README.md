# 🏎️ F1 Pit Stop Optimal Lap Predictor

A supervised machine learning system that predicts the optimal lap for an F1 driver to take a pit stop, trained on real telemetry data from the 2023 and 2024 Formula 1 seasons.

---

## 📌 Overview

Pit stop timing is one of the most consequential decisions in motorsport. This project frames it as a **binary classification problem**: given a snapshot of race conditions at any lap, can a model reliably classify whether that lap is an optimal pit stop lap?

We trained and compared three classifiers on **51,472 lap records** spanning **46 Grand Prix events**. The final model — XGBoost — achieved a **ROC-AUC of 0.994** and correctly identified both of Verstappen's pit windows at the 2023 Bahrain GP with near-1.0 probability spikes and zero false positives across 57 laps.

---

## 📊 Results

| Model | ROC-AUC | F1 (pit class) | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.906 | ~0.26 | ~0.16 | ~0.84 |
| Random Forest | 0.994 | ~0.80 | ~0.76 | ~0.87 |
| **XGBoost (selected)** | **0.994** | **~0.82** | **~0.84** | **~0.84** |

XGBoost confusion matrix on 10,300 test samples:
- ✅ 9,768 true negatives
- ✅ 281 true positives
- ❌ 54 false positives
- ❌ 52 false negatives

---

## 🔍 Key Finding: Feature Importance

The single most powerful predictor is **`lap_time_delta`** — the deviation of the current lap time from a 3-lap rolling average — with an importance score of **0.521**, over 5× the next feature.

| Feature | Importance |
|---|---|
| `lap_time_delta` | 0.521 |
| `lap_time_seconds` | 0.096 |
| `tyre_age_squared` | 0.070 |
| `laps_since_last_pit` | 0.050 |
| `stint_number` | 0.045 |

This quantitatively confirms what F1 strategists know intuitively: the moment a driver's lap time diverges from their recent average is when the tyres are going off.

---

## 🗂️ Dataset

- **Source:** [FastF1 Python library](https://docs.fastf1.dev/) (official F1 timing feed)
- **Coverage:** 2023 & 2024 seasons — 46 Grand Prix races
- **Size:** 51,472 lap records, 1,792 pit stop laps (3.48% positive class)
- **Features engineered:** 15

### Engineered Features

| Feature | Description |
|---|---|
| `lap_time_seconds` | Lap time converted to seconds |
| `rolling_avg_lap_time` | 3-lap rolling mean per driver per race |
| `lap_time_delta` | Current lap time minus rolling average |
| `TyreLife` | Tyre age in laps |
| `tyre_age_squared` | Non-linear tyre degradation proxy |
| `compound_encoded` | Ordinal encoding (SOFT=0 … WET=4) |
| `race_progress` | Lap number / total laps |
| `stint_number` | Cumulative pit count per driver |
| `laps_since_last_pit` | Laps elapsed in current stint |
| `position_norm` | Position / 20 |
| `position_change` | Lap-over-lap position delta |
| `is_front_runner` | 1 if Position ≤ 5 |
| `driver_encoded` | Label-encoded driver ID |
| `team_encoded` | Label-encoded team ID |
| `racename_encoded` | Label-encoded race name |

---

## 🛠️ Tech Stack

- **Python 3.14**
- **FastF1** — telemetry and lap data
- **pandas / numpy** — data manipulation
- **scikit-learn** — preprocessing, Logistic Regression, Random Forest, metrics
- **imbalanced-learn** — SMOTE oversampling
- **XGBoost** — final classifier
- **matplotlib / seaborn** — visualisations (dark F1 aesthetic)
- **joblib** — model serialisation

---

## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/cdevansh043/f1-pit-stop-predictor.git
cd f1-pit-stop-predictor
```

### 2. Install dependencies

```bash
pip install fastf1 pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn joblib
```

> **macOS note:** XGBoost requires OpenMP. Run `brew install libomp` if you hit a `libxgboost.dylib` error.

### 3. Run the pipeline in order

```bash
python main.py           # Data collection + feature engineering → f1_features.csv
python preprocessing.py  # Scaling + SMOTE → processed_data.pkl
python train.py          # Train all 3 models → model_results.pkl
python visualize.py      # Generate all 4 figures → fig1–fig4 .png files
```

> **First run:** `main.py` downloads ~46 races of F1 telemetry via FastF1 (this takes time and ~2GB of cache). Subsequent runs load from cache instantly.

---

## 📁 Project Structure

```
f1-pit-stop-predictor/
│
├── main.py               # Data collection & feature engineering
├── preprocessing.py      # Train/test split, StandardScaler, SMOTE
├── train.py              # Model training & evaluation
├── visualize.py          # All 4 result figures
│
├── cache/                # FastF1 telemetry cache (auto-created, gitignored)
├── f1_pit_stop_data.csv  # Raw collected data (auto-created)
├── f1_features.csv       # Engineered features (auto-created)
├── processed_data.pkl    # Scaled + SMOTE data (auto-created)
├── model_results.pkl     # Trained model objects + metrics (auto-created)
├── scaler.pkl            # Fitted StandardScaler (auto-created)
│
├── fig1_model_comparison.png
├── fig2_feature_importance.png
├── fig3_lap_timeline.png
└── fig4_probability_distributions.png
```

---

## 📈 Visualisations

**Figure 1 — Model Comparison Dashboard**
ROC-AUC scores, F1/Precision/Recall grouped bars, overlaid ROC curves, and confusion matrices for all three models.

**Figure 2 — Feature Importance**
XGBoost gain-based feature importance revealing the dominance of `lap_time_delta`.

**Figure 3 — Verstappen, Bahrain 2023**
Three-panel timeline: lap time vs rolling average, predicted pit probability per lap, and tyre life coloured by compound. The model spikes to near-1.0 on laps 14 and 36 — the exact pit laps — with zero false positives.

**Figure 4 — Probability Distributions**
Average predicted pit probability bucketed by tyre age and race progress. The race progress chart shows peak pit probability in the 36–71% window, consistent with classical F1 two-stop strategy.

---

## ⚠️ Limitations

- Covers dry-weather Grand Prix races only (no Sprint races, no wet weather modelling)
- Target variable reflects **observed** pit behaviour, not theoretically optimal pits
- No safety car / VSC feature — a major real-world pit trigger
- No competitor awareness (undercut/overcut scenarios not modelled)
- Not a real-time system — FastF1 data requires post-session processing

---

## 🔮 Future Work

- Add `is_safety_car` flag using FastF1 `TrackStatus` data
- Engineer gap-to-rival features for undercut/overcut detection
- Integrate weather data (track temperature, rain probability)
- Replace tabular classifier with an LSTM for sequence modelling
- Connect to the F1 Live Timing API for race-day inference
- Threshold optimisation via precision-recall curve analysis

---

## 👥 Authors

- **Devansh Chauhan**
- **Rao Arham Aziz**

---

## 📚 References

- [FastF1 Documentation](https://docs.fastf1.dev/)
- Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. KDD '16.
- Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR.
- Breiman (2001). Random Forests. Machine Learning.
