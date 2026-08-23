"""
Machine Learning Module
-------------------------------------------
Trains predictive models (Ridge and Random Forest) using the V3B Statcast feature set.
Utilizes strict purged walk-forward validation to prevent data leakage. For any given
snapshot date 'T', the model is trained exclusively on historical observations where
the 28-day forward return window completely finished prior to 'T' (future_end_date < T).
"""

import pandas as pd
import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# V3B Features (Point-in-Time Fundamentals)
V3B_FEATURES = [
    'xwoba_long', 'xwoba_gap_long', 'barrel_pct_long', 'hardhit_pct_long',
    'xwoba_season', 'xwoba_gap_season', 'barrel_pct_season', 'hardhit_pct_season', 'pa_season',
    'xwoba_t30', 'xwoba_gap_t30', 'barrel_pct_t30', 'hardhit_pct_t30', 'pa_t30',
    'active_30d', 'days_since_game', 'price_median_30d', 'sales_count_30d', 'price_dispersion_30d',
    'k_pct_t30', 'bb_pct_t30', 'avg_ev_t30', 'max_ev_t30', 'sweet_spot_pct_t30', 
    'xba_gap_t30', 'xslg_gap_t30',
    'xwoba_vs_long', 'barrel_vs_long', 'hardhit_vs_long'
]

def purged_time_series_split(df, features, target_col='target_28d_log_return', min_train_rows=50):
    """
    Runs a purged walk-forward validation loop to generate out-of-sample predictions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The complete point-in-time modeling dataset with historical sales data, 
        features, and target variables.
    features : list
        List of string column names of the features to be used by the model.
    target_col : str, default 'target_28d_log_return'
        The forward return target variable to predict.
    min_train_rows : int, default 50
        The minimum number of fully matured historical rows required before 
        the model can make predictions for a test week.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame of all usable out-of-sample test weeks, appended with their 
        respective Random Forest ('pred_rf') and Ridge ('pred_ridge') model predictions.
        Returns an empty DataFrame if no weeks meet the minimum training row requirement.
    """
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    df['future_end_date'] = pd.to_datetime(df['future_end_date'])
    
    unique_test_dates = df['snapshot_date'].drop_duplicates().sort_values().tolist()
    out_of_sample_preds = []
    
    # Random Forest Pipeline
    rf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('model', RandomForestRegressor(
            n_estimators=500, 
            max_depth=4, 
            min_samples_leaf=4, 
            max_features=0.7, 
            random_state=42, 
            n_jobs=-1
        ))
    ])
    
    # Ridge Pipeline (Requires StandardScaler so metrics are normalized and applied evenly)
    ridge_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ])
    
    print("\nStarting Purged Walk-Forward Validation...")
    print(f"Minimum fully matured training rows required = {min_train_rows}")
    
    for test_date in unique_test_dates: 
        
        train_mask = df['future_end_date'] < test_date
        test_mask = df['snapshot_date'] == test_date
        
        train_df = df[train_mask].dropna(subset=[target_col])
        test_df = df[test_mask].dropna(subset=[target_col]) 
        
        if len(train_df) < min_train_rows or len(test_df) == 0:
            continue
            
        X_train = train_df[features]
        y_train = train_df[target_col]
        X_test = test_df[features]
        
        # Train and Predict Random Forest
        rf_pipeline.fit(X_train, y_train)
        rf_preds = rf_pipeline.predict(X_test)
        
        # Train and Predict Ridge Baseline
        ridge_pipeline.fit(X_train, y_train)
        ridge_preds = ridge_pipeline.predict(X_test)
        
        eval_df = test_df.copy()
        eval_df['pred_rf'] = rf_preds
        eval_df['pred_ridge'] = ridge_preds
        out_of_sample_preds.append(eval_df)
        
    if out_of_sample_preds:
        final_res = pd.concat(out_of_sample_preds, ignore_index=True)
        print(f"Generated {len(final_res['snapshot_date'].unique())} usable test weeks ({len(final_res)} rows).")
        return final_res
        
    return pd.DataFrame()

if __name__ == "__main__":
    
    os.makedirs('data', exist_ok=True)
    dataset_path = 'data/model_dataset.csv'
    output_path = 'data/ml_predictions.csv'
    
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        
        # Features validation check
        missing_features = [f for f in V3B_FEATURES if f not in df.columns]
        if missing_features:
            raise ValueError(f"Model dataset is missing required V3B features: {missing_features}")
            
        preds_df = purged_time_series_split(df, V3B_FEATURES, min_train_rows=50)
            
        if not preds_df.empty:
            preds_df.to_csv(output_path, index=False)
            print(f"\nSuccess! Saved all out-of-sample predictions to '{output_path}'")
        else:
            print("\nError! No predictions generated. Dataset may not have enough usable rows.")
            sys.exit(1)
            
    else:
        print(f"\nError! '{dataset_path}' not found. Confirm that build_features.py has been run.")
        sys.exit(1)
