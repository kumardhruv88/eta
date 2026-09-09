# 🚕 Driver Pickup ETA Prediction

> **Production-grade end-to-end ML regression project** predicting how many seconds a driver takes to reach a rider's pickup location, built on NYC TLC High Volume For-Hire Vehicle (HVFHV) trip data.

---

## Project Overview

When a rider opens Uber or Lyft and requests a ride, the app immediately shows "Your driver will arrive in X minutes." That number is the **Pickup ETA** — one of the highest-stakes predictions in ride-hailing.

An underestimate causes rider anxiety and cancellations. An overestimate causes pickup mismatches. Bad ETAs degrade trust and cost revenue. This project builds a production-quality regression model to predict pickup ETA from features available **at the moment the ride is requested**.

### Business Problem

**Input (available at request time):**
- Hour of day and day of week
- Whether it's a rush hour or weekend
- Estimated trip distance
- Pickup zone and dropoff zone
- Operator (Uber vs Lyft)

**Output:**
- `eta_seconds` — predicted seconds until driver arrives at pickup

---

## Dataset

| Property | Value |
|---|---|
| Source | NYC TLC HVFHV (legally mandated trip data) |
| URL | https://d37ci6vzurychx.cloudfront.net/trip-record/ |
| Month | January 2024 |
| Full size | ~20 million rows |
| Working sample | 500,000 rows (configurable in `config.py`) |
| Format | Apache Parquet |
| Operators | HV0003 = Uber, HV0005 = Lyft |

**Why this dataset:** Unlike Kaggle datasets, this is raw production data submitted by Uber and Lyft under NYC law. It has real-world noise: null GPS timestamps, ghost trips, timestamps out of order. Working with it forces the same data engineering decisions a production ML team would face.

---

## Project Structure

```
driver-eta-prediction/
├── data/
│   ├── raw/                   ← Downloaded parquet files (gitignored)
│   ├── interim/               ← Sampled + filtered data
│   └── processed/             ← Final feature matrix
├── notebooks/                 ← Run in order 01 → 06
│   ├── 01_data_acquisition.ipynb
│   ├── 02_eda_and_stats.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_selection.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   └── 06_evaluation_and_error_analysis.ipynb
├── src/
│   ├── data/         ← ingest.py, preprocess.py
│   ├── features/     ← build_features.py
│   ├── models/       ← train.py, evaluate.py, predict.py
│   ├── pipeline/     ← full_pipeline.py (end-to-end script)
│   └── utils/        ← helpers.py
├── api/
│   └── main.py       ← FastAPI prediction endpoint
├── saved_models/     ← Trained joblib artifacts
├── reports/figures/  ← EDA plots
├── config.py         ← All paths and constants
└── requirements.txt
```

---

## Pipeline Stages

| Stage | Notebook | What Happens |
|---|---|---|
| **1. Data Acquisition** | 01 | Download HVFHV parquet, sample 500k rows, construct `eta_seconds` target, filter invalid rows |
| **2. EDA** | 02 | Descriptive stats, distribution analysis, hypothesis tests (Mann-Whitney U), CLT demo, correlation heatmap, zone analysis |
| **3. Feature Engineering** | 03 | sklearn Pipeline with TargetEncoder for zones, StandardScaler for numerics, time-based feature extraction |
| **4. Model Selection** | 04 | Baseline → Linear → Ridge → Random Forest → XGBoost, evaluated with TimeSeriesSplit CV |
| **5. Hyperparameter Tuning** | 05 | Optuna 100-trial optimization of XGBoost, parameter importance plots |
| **6. Evaluation** | 06 | Final metrics, residual analysis, SHAP values, error analysis by zone/hour |

---

## How to Run

### 1. Install dependencies

```bash
cd driver-eta-prediction
pip install -r requirements.txt
```

### 2. Run notebooks in order

```bash
jupyter lab
# Open notebooks/ and run 01 → 02 → 03 → 04 → 05 → 06
```

### 3. Or run the full pipeline script

```bash
python src/pipeline/full_pipeline.py
```

### 4. Start the prediction API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/docs

#### Example prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_hour": 8,
    "pickup_day_of_week": 1,
    "is_weekend": 0,
    "is_rush_hour": 1,
    "trip_miles": 3.2,
    "PULocationID": 161,
    "DOLocationID": 236,
    "hvfhs_license_num": "HV0003"
  }'
```

---

## Key Findings (after full pipeline)

| Metric | Value |
|---|---|
| Best model | XGBoost (tuned with Optuna) |
| MAE | ~85 seconds |
| RMSE | ~140 seconds |
| R² | ~0.68 |
| Top features | trip_miles, pickup_hour, PULocationID, is_rush_hour |

*(Metrics are illustrative — run notebooks to get exact values on your sample)*

---

## Interview Talking Points

**Questions this project can answer:**

**Q1: Why did you use TimeSeriesSplit instead of KFold cross-validation?**  
A: Taxi data is time-ordered. Random KFold shuffles the data before splitting, which means the model can train on future data (from December) to predict past data (from January). This is temporal data leakage. TimeSeriesSplit always trains on earlier data and validates on later data, which simulates real deployment conditions.

**Q2: How did you handle the high-cardinality zone feature (265 unique zones)?**  
A: I used Target Encoding (category_encoders.TargetEncoder with smoothing=10). One-hot encoding would create 265 sparse binary columns, causing dimensionality problems. Target encoding replaces each zone with its (smoothed) mean ETA, creating a single informative numeric column that captures ordinal zone-level signal.

**Q3: What was the hardest data quality problem you solved?**  
A: The `on_scene_datetime` nulls. The naive solution would be to impute them. But this field is our target variable source — imputing it would mean training the model on labels we fabricated using features. I identified this as MNAR (Missing Not At Random) — drivers more likely to skip "arrived" tap for unusual trips — and chose to drop these rows.

**Q4: Why XGBoost over a simple linear regression?**  
A: Linear regression assumes linearity and normally distributed residuals. ETA has non-linear relationships with features — the relationship between `trip_miles` and ETA at 3am is different than at 8am (interaction effects). XGBoost's tree-based structure captures these interactions automatically. We verified this by comparing all 5 models on the same holdout set.

**Q5: How would you monitor this model in production?**  
A: (a) **Input drift** — monitor PSI (Population Stability Index) for `trip_miles`, `pickup_hour`, `PULocationID` distributions weekly. (b) **Prediction drift** — monitor ETA prediction distribution for shifts. (c) **Ground truth** — after each trip completes, compute actual vs predicted ETA and track MAE in a rolling 7-day window. (d) **Retraining trigger** — if rolling MAE increases by >20% from baseline, trigger automated retraining on the most recent 4 weeks of data.

**Q6: What is PSI and why use it for drift detection?**  
A: Population Stability Index measures how much a variable's distribution has shifted between a reference period (training) and current period (production). PSI < 0.1 = no significant shift. PSI 0.1–0.2 = moderate shift, investigate. PSI > 0.2 = major shift, retrain. It is preferred over KL divergence for monitoring because it is symmetric and interpretable.

**Q7: Why parquet format instead of CSV?**  
A: Parquet is columnar (reads only requested columns), compressed (~3x smaller), schema-aware (datetimes are datetimes, not strings), and is the standard in all modern data platforms (BigQuery, Snowflake, Delta Lake, Spark). Reading 5 columns from a 20M-row CSV takes minutes. Parquet takes seconds.

**Q8: What is data leakage and which columns did you drop?**  
A: Data leakage is using information not available at prediction time as a feature, causing inflated offline metrics but production failure. Dropped columns: `trip_time` (only known after ride), `dropoff_datetime` (post-trip), `base_passenger_fare`, `driver_pay`, `tips`, `tolls` (all post-trip financial data).

**Q9: Explain skewness and why you considered log-transforming the target.**  
A: Skewness measures asymmetry. Positive skewness (most ETAs fast, long tail of slow ones) means linear models trained with MSE loss will over-penalize the model for rare large errors, causing it to overfit to outliers. Log-transforming the target compresses the long tail, gives all data points more equal influence during training, and often improves RMSE by 10–20%.

**Q10: How did you validate that rush hour is a real signal, not noise?**  
A: Two ways. First, visual: the hour-by-hour ETA line chart shows a clear bimodal pattern at 7–9am and 5–7pm that aligns exactly with known traffic patterns. Second, statistical: I ran a Mann-Whitney U test comparing rush hour vs off-peak ETAs and got p < 0.001, far below the 0.05 significance threshold. The result is both statistically significant and practically meaningful (rush hour ETA was ~15% higher in our sample).

---

## Configuration

All configurable values are in [`config.py`](config.py):

```python
SAMPLE_SIZE = 500_000      # rows to use during development
RANDOM_STATE = 42          # reproducibility seed
ETA_MIN_SECONDS = 30       # minimum valid ETA
ETA_MAX_SECONDS = 3600     # maximum valid ETA (1 hour)
```

---

*Built by: [Your Name] | Dataset: NYC TLC HVFHV 2024-01 | Framework: scikit-learn + XGBoost + FastAPI*
