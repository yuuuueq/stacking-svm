"""
This script is used to evaluate the performance of multiple machine learning models
based on different performance indicators. It reads performance data from CSV files,
normalizes the data, calculates scores for each model, and finally sorts and prints
the mean scores of all models.

Performance Indicators:
- R2: Coefficient of determination
- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error

Models:
A variety of single models and stacking models are included, such as XGBoost, LightGBM,
CatBoost, AdaBoost, and stacking models with different base models and final estimators.

Input:
- CSV files named after performance indicators (e.g., R2.csv, MSE.csv) in the './indicators' directory.
  Each file contains performance data for different models and MMSI numbers.

Output:
- Mean scores for each model, printed in the console.
- Sorted results of mean scores in descending order, printed in the console.

Author: Yueqi Zhang
Date: 2025-04
"""

import pandas as pd
import numpy as np

# Performance indicators
indicators = ["R2", "MSE", "RMSE", "MAPE"]

# Models
models = ["XGBoost", "LightGBM", "CatBoost", "RF", "DT", "GBDT", "SVM", "Ridge", "KNN", "MLR", "ET", "HGBT",
          "Stacking-Ridge",
          "Stacking-ET", "Stacking-RF", "Stacking-XGB",
          "Stacking-CatB", "Stacking-HGBT", "Stacking-LGBM",
          "Stacking-DT", "Stacking-GBDT", "Stacking-SVM",
          "Stacking-KNN", "Stacking-MLR"]

# Weights for each indicator
weights = {"R2": 5, "MSE": 5, "RMSE": 5, "MAPE": 5}

# Scores for each model
scores = {}
for model in models:
    scores[model] = []

mmsis = ["215126000", "215131000", "215189000", "215240000", "215577000"]

# Normalization function
def normalize(lst):
    min_val = min(lst)
    max_val = max(lst)
    normalized_lst = [(x - min_val) / (max_val - min_val) for x in lst]
    return normalized_lst

for mmsi in mmsis:
    R2s = []
    MSEs = []
    RMSEs = []
    MAPEs = []
    for indicator in indicators:
        df = pd.read_csv(f"./indicators/{indicator}.csv")
        data = df.iloc[:, 1:]
        data.index = list(df["MMSI"])

        if indicator == "R2":
            for model in models:
                R2s.append(data.loc[int(mmsi), model])

        if indicator == "MSE":
            for model in models:
                MSEs.append(data.loc[int(mmsi), model])

        if indicator == "RMSE":
            for model in models:
                RMSEs.append(data.loc[int(mmsi), model])

        if indicator == "MAPE":
            for model in models:
                MAPEs.append(data.loc[int(mmsi), model])

    # Normalize some data
    normalized_MSEs = normalize(MSEs)
    normalized_RMSEs = normalize(RMSEs)

    df_new = pd.DataFrame([R2s, normalized_MSEs, normalized_RMSEs, MAPEs])
    df_new.columns = models
    df_new.index = ["R2", "MSE", "RMSE", "MAPE"]

    # Calculate scores
    for index, model in enumerate(models):
        current_score = (weights["R2"] * df_new.loc["R2", model]
                         + weights["MSE"] * (1 - df_new.loc["MSE", model])
                         + weights["RMSE"] * (1 - df_new.loc["RMSE", model])
                         + weights["MAPE"] * (1 - df_new.loc["MAPE", model])) / 20
        scores[model].append(current_score)

# Calculate mean scores
mean_scores = {}
for model in scores.keys():
    mean_scores[model] = sum(scores[model]) / len(scores[model])

# Print mean scores
for model in mean_scores.keys():
    print(f"{model}: {mean_scores[model]}")

# Sort mean scores in descending order
sorted_mean_scores = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)

# Print sorted results
print("\n=========== Sorted Results ===========")
for index, item in enumerate(sorted_mean_scores):
    print(f"{index + 1}. {item[0]}: {item[1]}")
