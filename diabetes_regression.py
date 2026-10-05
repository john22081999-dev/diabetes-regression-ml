import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    max_error,
    mean_absolute_percentage_error,
)

os.makedirs("plots/actual_vs_predicted", exist_ok=True)
os.makedirs("plots/shap_summary", exist_ok=True)
os.makedirs("plots/shap_waterfall", exist_ok=True)


# Load dataset

diabetes_data = load_diabetes(as_frame=True, scaled=False)
df = diabetes_data.frame  

# 1.1 Print the value range of one column 
print("Range of 'bmi':", df["bmi"].min(), "-", df["bmi"].max())

# 1.2 Check for NaN values and drop affected rows if any
print("NaN count per column:\n", df.isna().sum())
df = df.dropna().reset_index(drop=True)

# 1.3 Histogram of target column
plt.figure(figsize=(6, 4))
plt.hist(df["target"], bins=30, color="steelblue", edgecolor="black")
plt.title("Histogram of target (disease progression after 1 year)")
plt.xlabel("target")
plt.ylabel("frequency")
plt.tight_layout()
plt.savefig("plots/target_histogram.png")
plt.close()


# Split features / target and apply Min-Max normalization

feature_cols = [c for c in df.columns if c != "target"]
X_raw = df[feature_cols].values
y_raw = df["target"].values.reshape(-1, 1)

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X = x_scaler.fit_transform(X_raw)
y = y_scaler.fit_transform(y_raw).ravel()


SEED = 42

# KFold split 

N_FOLDS = 6
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


# Results container

csv_data = []
csv_data.append(["Model", "Set", "Fold",
                  "Max_n", "RMSE_n", "MAE_n", "MAPE_n",
                  "Max", "RMSE", "MAE", "MAPE"])


gp_kernels = [
    ConstantKernel() * RBF(length_scale=l) + WhiteKernel(noise_level=n)
    for l in [0.5, 1.0, 2.0]
    for n in [1e-5, 1e-3, 1e-1]
]

model_configs = {
    "RandomForest": {
        "estimator": RandomForestRegressor(random_state=SEED),
        "param_distributions": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [None, 3, 5, 8, 12],
            "min_samples_split": [2, 4, 6, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "n_iter": 15,
    },
    "GaussianProcess": {
        "estimator": GaussianProcessRegressor(random_state=SEED, normalize_y=False),
        "param_distributions": {
            "alpha": [1e-10, 1e-5, 1e-3, 1e-2, 1e-1],
            "kernel": gp_kernels,
        },
        "n_iter": 8,  # GP is slow, use fewer search iterations
    },
    "SVR": {
        "estimator": SVR(),
        "param_distributions": {
            "kernel": ["rbf", "linear", "poly"],
            "C": [0.1, 1, 10, 50, 100],
            "epsilon": [0.01, 0.05, 0.1, 0.2],
            "gamma": ["scale", "auto"],
        },
        "n_iter": 15,
    },
    "KNN": {
        "estimator": KNeighborsRegressor(),
        "param_distributions": {
            "n_neighbors": [3, 5, 7, 9, 11, 15],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
        "n_iter": 12,
    },
}

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mx = max_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mx, rmse, mae, mape



# Main loop: for every fold, train every model

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    for model_name, cfg in model_configs.items():
        print(f"Fold {fold_idx} | Training {model_name} ...")

        search = RandomizedSearchCV(
            estimator=cfg["estimator"],
            param_distributions=cfg["param_distributions"],
            n_iter=cfg["n_iter"],
            cv=3,
            random_state=SEED,
            n_jobs=-1,
            scoring="neg_root_mean_squared_error",
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # Predictions 
        y_train_pred_n = best_model.predict(X_train)
        y_test_pred_n = best_model.predict(X_test)

        # Denormalize back to the original scale
        y_train_true = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
        y_test_true = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_train_pred = y_scaler.inverse_transform(y_train_pred_n.reshape(-1, 1)).ravel()
        y_test_pred = y_scaler.inverse_transform(y_test_pred_n.reshape(-1, 1)).ravel()

        # Metrics 
        max_n_tr, rmse_n_tr, mae_n_tr, mape_n_tr = compute_metrics(y_train, y_train_pred_n)
        max_n_te, rmse_n_te, mae_n_te, mape_n_te = compute_metrics(y_test, y_test_pred_n)

        # Metrics - original scale
        mx_tr, rmse_tr, mae_tr, mape_tr = compute_metrics(y_train_true, y_train_pred)
        mx_te, rmse_te, mae_te, mape_te = compute_metrics(y_test_true, y_test_pred)

        csv_data.append([model_name, "Train", fold_idx,
                          max_n_tr, rmse_n_tr, mae_n_tr, mape_n_tr,
                          mx_tr, rmse_tr, mae_tr, mape_tr])
        csv_data.append([model_name, "Test", fold_idx,
                          max_n_te, rmse_n_te, mae_n_te, mape_n_te,
                          mx_te, rmse_te, mae_te, mape_te])

        # Actual vs Predicted plot 
        plt.figure(figsize=(5, 5))
        plt.scatter(y_test_true, y_test_pred, alpha=0.7, label=f"{model_name} predictions")
        lims = [min(y_test_true.min(), y_test_pred.min()),
                max(y_test_true.max(), y_test_pred.max())]
        plt.plot(lims, lims, "r--", label="Ideal")
        plt.xlabel("Actual Output")
        plt.ylabel("Predicted Output")
        plt.title(f"Actual vs Predicted - {model_name} - Fold {fold_idx} (Test)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/actual_vs_predicted/{model_name}_fold{fold_idx}_test.png")
        plt.close()

        # SHAP 
        try:
            background_size = min(50, len(X_train))
            background = shap.sample(X_train, background_size, random_state=SEED)
            explainer = shap.KernelExplainer(best_model.predict, background)

            sample_size = min(30, len(X_test))
            shap_values = explainer(X_test[:sample_size])

            plt.figure()
            shap.summary_plot(
                shap_values, X_test[:sample_size],
                feature_names=feature_cols, show=False
            )
            plt.title(f"SHAP Summary - {model_name} - Fold {fold_idx}")
            plt.tight_layout()
            plt.savefig(f"plots/shap_summary/{model_name}_fold{fold_idx}_summary.png")
            plt.close()

            # Waterfall plots 
            for i in [0, 1]:
                plt.figure()
                shap.plots.waterfall(shap_values[i], show=False)
                plt.tight_layout()
                plt.savefig(
                    f"plots/shap_waterfall/{model_name}_fold{fold_idx}_example{i}.png"
                )
                plt.close()
        except Exception as e:
            print(f"SHAP failed for {model_name} fold {fold_idx}: {e}")


# Save results to CSV 

results_df = pd.DataFrame(csv_data[1:], columns=csv_data[0])
results_df.to_csv("results.csv", index=False)
print("Done. Results saved to results.csv")