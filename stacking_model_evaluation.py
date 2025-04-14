"""
Stacking Model for Ship Speed Prediction

Features:
- Data preprocessing and standardization
- Stacking ensemble model construction
- Performance metrics calculation (R2, MSE, RMSE, MAPE)
- Training time measurement
- Results saving to CSV files

Dataset:
- Source: target_{mmsi}.xlsx files
- Ships: 5 ships with different MMSI numbers
- Features: 8 environmental and navigational parameters
- Target: Ship speed (filtered range: 12-30 knots)
- Train/Test split: 70/30

Author: Yueqi Zhang
Date: 2025-04
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error
import time

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


def create_stacking_model(final_estimator):
    """Create stacking ensemble model"""
    base_estimators = [
        ('et', ExtraTreesRegressor()),
        ('rf', RandomForestRegressor()),
        ('catb', CatBoostRegressor(verbose=False)),
        ('xgb', XGBRegressor()),
        ('lgbm', LGBMRegressor(verbose=-1)),
        ('hgbt', HistGradientBoostingRegressor()),
    ]
    return StackingRegressor(
        estimators=base_estimators,
        final_estimator=final_estimator
    )


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train and evaluate the stacking model"""
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


def update_metrics_files(metrics_collection):
    """Update all metrics files with new values"""
    for final_estimator_name, estimator_metrics in metrics_collection.items():
        base_model_names = 'Stacking'
        model_name = f'{base_model_names}-{final_estimator_name})'
        for metric_name, values in estimator_metrics.items():
            df = pd.read_csv(f"./indicators/{metric_name}.csv")
            df[model_name] = values
            df.to_csv(f"./indicators/{metric_name}.csv", index=False)


def main():
    """Main function to run the stacking model evaluation"""
    mmsis = ["215126000", "215131000", "215189000", "215240000", "215577000"]
    final_estimators = {
        'XGB': XGBRegressor(),
        'LGBM': LGBMRegressor(verbose=-1),
        'CatB': CatBoostRegressor(verbose=False),
        'RF': RandomForestRegressor(),
        'ET': ExtraTreesRegressor(),
        'DT': DecisionTreeRegressor(),
        'GBDT': GradientBoostingRegressor(),
        'SVM': SVR(),
        'KNN': KNeighborsRegressor(),
        'MLR': LinearRegression(),
        'HGBT': HistGradientBoostingRegressor(),
        'Ridge': Ridge()
    }
    all_metrics_collection = {name: {metric: [] for metric in ['R2', 'MSE', 'RMSE', 'MAPE', 'Running_time']}
                              for name in final_estimators.keys()}

    for mmsi in mmsis:
        print(f"Processing MMSI: {mmsi}")
        X_train, X_test, y_train, y_test = load_and_preprocess_data(mmsi)

        for estimator_name, estimator in final_estimators.items():
            model = create_stacking_model(estimator)
            metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

            for metric_name, value in metrics.items():
                all_metrics_collection[estimator_name][metric_name].append(value)

    update_metrics_files(all_metrics_collection)
    print("Evaluation completed and results saved.")


if __name__ == "__main__":
    main()
 