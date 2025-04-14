"""
This script is designed to build a stacking regression model for ship speed prediction
and use SHAP (SHapley Additive exPlanations) values to analyze the importance and
impact of features on the model's output.

It performs the following steps:
1. Loads ship data from Excel files for multiple ships identified by MMSI numbers.
2. Preprocesses the data, including filtering, renaming columns, and standardizing features.
3. Splits the data into training and test sets.
4. Defines base models and a meta-model to create a stacking regression model.
5. Trains the stacking model on the training data.
6. Uses SHAP to explain the model's predictions and generates various visualizations,
   such as summary plots, bar plots, heatmaps, dependency plots, and scatter plots.
7. Saves all the generated plots as PNG images in the specified directory.

Inputs:
- Excel files named 'target_{mmsi}.xlsx' in the '../data' directory, where {mmsi} is the ship's MMSI number.
  Each file should contain ship-related features and the target variable 'ShipSpeed'.

Outputs:
- Trained stacking regression model.
- Multiple SHAP visualization plots saved as PNG images in the '../figs/{mmsi}' directory,
  where {mmsi} is the ship's MMSI number.
  
Author: Yueqi Zhang
Date: 2025-04
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
import numpy as np
from joblib import dump, load
import matplotlib.pyplot as plt
import shap

np.seterr(divide='ignore')

mmsis = ["215126000", "215131000", "215189000", "215240000", "215577000"]

for mmsi in mmsis:
    # Load data from Excel file
    data = pd.read_excel(f'../data/target_{mmsi}.xlsx')

    # Filter data based on ship speed
    data = data[(data['ShipSpeed'] >= 12) & (data['ShipSpeed'] <= 30)]

    # Reset index
    data.reset_index(drop=True, inplace=True)

    # Rename columns
    data.rename(columns={'WindSpeed': 'Wind speed'}, inplace=True)
    data.rename(columns={'RelativeWindDirection': 'Relative wind direction'}, inplace=True)
    data.rename(columns={'WaveHeight': 'Wave height'}, inplace=True)
    data.rename(columns={'RelativeWaveDirection': 'Relative wave direction'}, inplace=True)
    data.rename(columns={'WavePeriod': 'Wave period'}, inplace=True)
    data.rename(columns={'CurrentSpeed': 'Current speed'}, inplace=True)
    data.rename(columns={'RelativeCurrentDirection': 'Relative current direction'}, inplace=True)
    data.rename(columns={'Course': 'Ship course'}, inplace=True)

    # Extract features and target variable
    X = data.iloc[:, :8].values
    y = data.iloc[:, -1].values

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define base models
    et_model = ExtraTreesRegressor()
    rf_model = RandomForestRegressor()
    xgb_model = XGBRegressor()
    catb_model = CatBoostRegressor(verbose=0)
    hgbt_model = HistGradientBoostingRegressor()
    lgbm_model = LGBMRegressor(verbose=-1)

    # Define meta-model
    meta_model = SVR()

    # Build stacking model
    stacking_model = StackingRegressor(
        estimators=[('et', et_model), ('rf', rf_model), ('xgb', xgb_model), ('catb', catb_model), ('hgbt', hgbt_model), ('lgbm', lgbm_model)],
        final_estimator=meta_model,
    )

    # Train stacking model
    stacking_model.fit(X_train, y_train)

    # Create SHAP explainer
    explainer = shap.KernelExplainer(stacking_model.predict, shap.sample(X_train, 100), feature_names=data.columns[:-1])

    # Calculate SHAP values
    shap_values = explainer(X_test)

    # Set global font to Times New Roman
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Times New Roman'

    # Set matplotlib to non-interactive mode
    plt.ioff()

    # 1. Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.yticks(fontsize=16, color='blue')
    plt.xticks(fontsize=16)
    plt.xlabel("SHAP value (impact on model output)", fontsize=16)
    plt.savefig(f'../figs/{mmsi}/shap_summary_plot_{mmsi}.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 2. Bar Plot
    plt.figure()
    shap.plots.bar(shap_values)
    plt.yticks(fontsize=16, color='blue')
    plt.xticks(fontsize=16)
    plt.xlabel("mean(|SHAP value|)", fontsize=16)
    plt.savefig(f'../figs/{mmsi}/shap_bar_plot_{mmsi}.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 3. Heatmap
    plt.figure()
    shap.plots.heatmap(shap_values)
    plt.yticks(fontsize=16, color='blue')
    plt.xticks(fontsize=16)
    plt.xlabel("Instances", fontsize=16)
    plt.savefig(f'../figs/{mmsi}/shap_heatmap_{mmsi}.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 4. Dependency Plots for Each Feature
    feature_names = data.columns[:-1]
    for i, feature in enumerate(feature_names):
        plt.figure()
        shap.dependence_plot(feature, shap_values.values, X_test, feature_names=feature_names, interaction_index=None, show=False)
        plt.xlabel(feature, fontsize=14)
        plt.ylabel('SHAP Value', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(f'../figs/{mmsi}/shap_dependence_{feature}_{mmsi}.png', bbox_inches='tight', dpi=300)
        plt.close()

    # 5. Scatter Plots for Each Feature
    for i, feature in enumerate(feature_names):
        plt.figure()
        shap.plots.scatter(shap_values[:, feature], color=shap_values, show=False)
        plt.xlabel(f'SHAP Value for {feature}', fontsize=14)
        plt.ylabel('Feature Value', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(f'../figs/{mmsi}/shap_scatter_{feature}_{mmsi}.png', bbox_inches='tight', dpi=300)
        plt.close()

    # Turn interactive mode back on if needed
    plt.ion()
    