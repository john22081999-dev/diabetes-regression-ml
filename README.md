# Diabetes Progression — Regression & Model Comparison

A machine-learning project that predicts one-year disease progression for
diabetes patients and compares four regression models, with SHAP-based
explainability of the predictions.

## Problem

Using the public **scikit-learn Diabetes dataset** (442 patients, 10 features:
age, sex, BMI, blood pressure and six blood serum measurements s1–s6), the goal
is to predict a continuous target representing quantitative disease progression
one year after baseline — and to understand *which features drive* each
prediction.

## Approach

- **Preprocessing:** load features in their original scale, check for missing
  values, and apply Min-Max normalization to both features and target (the
  target is denormalized before computing error metrics, so results are in real
  units).
- **Cross-validation:** 6-fold `KFold` (shuffled, fixed seed). Each model is
  trained only on the training folds and evaluated on both train and test
  subsets to inspect overfitting.
- **Models compared:**
  - Random Forest Regressor
  - Gaussian Process Regressor (RBF + White kernel)
  - Support Vector Regression (SVR)
  - K-Nearest Neighbors Regressor
- **Hyperparameter tuning:** `RandomizedSearchCV` on each model.
- **Metrics:** MSE, MAE, MAPE and max error, averaged across folds.
- **Explainability:** SHAP values, with per-model summary and waterfall plots
  explaining individual predictions.

## Results

The `plots/` folder contains the full evaluation output:

- `plots/target_histogram.png` — distribution of the target variable
- `plots/actual_vs_predicted/` — actual vs predicted scatter per model and fold
- `plots/shap_waterfall/` — SHAP explanations for individual predictions

Per-fold error metrics (MSE, MAE, MAPE, max error) for each model are printed to
the console when the script runs.

## Tech stack

- Python 3
- scikit-learn, pandas, numpy
- matplotlib (plots)
- SHAP (model explainability)

## How to run

1. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the script:
   ```
   python diabetes_regression.py
   ```
   Plots are written to the `plots/` folder and metrics are printed to the
   console.

## Notes

This project uses the built-in scikit-learn Diabetes dataset, so no external
data download is required. It began as a university machine-learning assignment
and has been cleaned up and documented as a standalone portfolio project.
