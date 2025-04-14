"""
Ship Speed Prediction Model Evaluation

This script implements and evaluates multiple machine learning models for ship speed prediction:
1. XGBoost: Extreme Gradient Boosting
2. LightGBM: Light Gradient Boosting Machine
3. CatBoost: Categorical Boosting
4. RF: Random Forest
5. ET: Extra Trees
6. DT: Decision Tree
7. GBDT: Gradient Boosting Decision Tree
8. SVM: Support Vector Machine
9. KNN: K-Nearest Neighbors
10. MLR: Multiple Linear Regression
11. HGBT: Histogram-based Gradient Boosting Trees
12. Ridge: Ridge Regression

Features:
- Data preprocessing and standardization
- Model training and evaluation
- Performance metrics:
  * R-squared (R2)
  * Mean Squared Error (MSE)
  * Root Mean Squared Error (RMSE)
  * Mean Absolute Percentage Error (MAPE)
  * Training time

Dataset:
- Source: target_{mmsi}.xlsx files
- Ships: 5 ships with different MMSI numbers
- Features: 8 environmental and navigational parameters
- Target: Ship speed (filtered range: 12-30 knots)
- Train/Test split: 70/30

Results:
- All metrics are saved in ./indicators/ directory
- Separate CSV files for each metric type

Author: Yueqi Zhang
Date: 2025-04
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error
import time
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, 
                            GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
np.seterr(divide='ignore')

def load_and_preprocess_data(mmsi):
    """Load and preprocess data for a specific ship"""
    data = pd.read_excel(f'./fusion_data/target_{mmsi}.xlsx')
    data = data[(data['ShipSpeed'] >= 12) & (data['ShipSpeed'] <= 30)].reset_index(drop=True)
    
    X = data.iloc[:, :8].values
    y = data.iloc[:, -1].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return train_test_split(X, y, test_size=0.3, random_state=42)

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train and evaluate a single model"""
    start_time = time.time()
    model.fit(X_train, y_train)
    running_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    
    return {
        'R2': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': root_mean_squared_error(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred),
        'Running_time': running_time
    }

def get_models():
    """Initialize all models with their configurations"""
    return {
        'XGBoost': XGBRegressor(),
        'LightGBM': LGBMRegressor(verbose=-1),
        'CatBoost': CatBoostRegressor(verbose=False),
        'RF': RandomForestRegressor(),
        'ET': ExtraTreesRegressor(),
        'DT': DecisionTreeRegressor(),
        'GBDT': GradientBoostingRegressor(),
        'SVM': SVR(kernel='rbf'),
        'KNN': KNeighborsRegressor(),
        'MLR': LinearRegression(),
        'HGBT': HistGradientBoostingRegressor(),
        'Ridge': Ridge()
    }

def update_metrics_files(metrics_collection):
    """Update all metrics files with new values"""
    for metric_name in ['R2', 'MSE', 'RMSE', 'MAPE', 'Running_time']:
        df = pd.read_csv(f"./indicators/{metric_name}.csv")
        for model_name, values in metrics_collection[metric_name].items():
            df[model_name] = values
        df.to_csv(f"./indicators/{metric_name}.csv", index=False)

def main():
    """Main function to evaluate all models"""
    mmsis = ["215126000", "215131000", "215189000", "215240000", "215577000"]
    models = get_models()
    
    # Initialize metrics collection
    metrics_collection = {
        metric: {model: [] for model in models.keys()}
        for metric in ['R2', 'MSE', 'RMSE', 'MAPE', 'Running_time']
    }
    
    # Evaluate each model on each ship's data
    for mmsi in mmsis:
        print(f"Processing MMSI: {mmsi}")
        X_train, X_test, y_train, y_test = load_and_preprocess_data(mmsi)
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
            
            for metric_name, value in metrics.items():
                metrics_collection[metric_name][model_name].append(value)
    
    # Update metrics files
    update_metrics_files(metrics_collection)
    print("All evaluations completed and results saved.")

if __name__ == "__main__":
    main()