"""
Live Prediction Module
---------------------------------------------
Generates model rankings for the latest available market snapshot.
The model is trained exclusively on historical data where the 28-day target outcome
is already known. The current/latest snapshot rows are isolated from the training set.
They do not include a target variable because their future has not happened yet.
Scaling and imputation are based only on the historical training data so that
current/future market conditions don't leak into the past.
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

V3B_FEATURES = [
    'xwoba_long', 'xwoba_gap_long', 'barrel_pct_long', 'hardhit_pct_long',
    'xwoba_season', 'xwoba_gap_season', 'barrel_pct_season', 'hardhit_pct_season', 'pa_season',
    'xwoba_t30', 'xwoba_gap_t30', 'barrel_pct_t30', 'hardhit_pct_t30', 'pa_t30',
    'active_30d', 'days_since_game', 'price_median_30d', 'sales_count_30d', 'price_dispersion_30d',
    'k_pct_t30', 'bb_pct_t30', 'avg_ev_t30', 'max_ev_t30', 'sweet_spot_pct_t30', 
    'xba_gap_t30', 'xslg_gap_t30',
    'xwoba_vs_long', 'barrel_vs_long', 'hardhit_vs_long'
]

MIN_TRAIN_ROWS = 50

def generate_live_predictions(df, features, target_col='target_28d_log_return'):
    """
    Trains models on all valid historical data and predicts future returns 
    for the most recent/current cross-sectional snapshot.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The complete point-in-time dataset containing all historical sales 
        and current snapshot.
    features : list
        List of string column names of the features used by the model.
    target_col : str, default 'target_28d_log_return'
        The forward return target variable to train on.
        
    Returns:
    --------
    pd.DataFrame
        A ranked DataFrame of the latest cards with their live 
        Random Forest and Ridge predictions.
    """
    
    df = df.copy()
    
    # Ensure date formatting for temporal splitting
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    df['future_end_date'] = pd.to_datetime(df['future_end_date'])
    
    # Identify present day (most recent available snapshot for prediction)
    latest_date = df['snapshot_date'].max()
    
    # Isolate live prediction universe (cards without a known 28-day return)
    live_mask = (df['snapshot_date'] == latest_date) & (df[target_col].isna())
    live_df = df[live_mask].copy()
    
    # Isolate historical training universe (cards with a mature, known 28-day return)
    mature_mask = (df['future_end_date'] < latest_date) & (df[target_col].notna())
    train_df = df[mature_mask].copy()
    
    print("\nLive Predictions Initiated:")
    print(f"\nLive Snapshot Date      : {latest_date.strftime('%Y-%m-%d')}")
    print(f"Cards to Predict        : {len(live_df)}")
    print(f"Mature Historical Rows  : {len(train_df)}")
    
    if len(train_df) < MIN_TRAIN_ROWS:
        raise ValueError(f"Insufficient historical data ({len(train_df)} rows). Require at least {MIN_TRAIN_ROWS}.")
    if len(live_df) == 0:
        raise ValueError("No live prediction rows found for the latest snapshot. Check target isolation rules.")
        
    X_train = train_df[features]
    y_train = train_df[target_col]
    X_live = live_df[features]
    
    # Create Random Forest pipeline
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
    
    # Create Ridge Baseline pipeline
    ridge_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ])
    
    # Train models on all available history
    print("\nTraining Random Forest Full History Model...")
    rf_pipeline.fit(X_train, y_train)
    
    print("Training Ridge Baseline Full History Model...")
    ridge_pipeline.fit(X_train, y_train)
    
    # Generate live predictions for the unknown future (log terms)
    print("Generating Live Predictions...")
    live_df['pred_rf'] = rf_pipeline.predict(X_live)
    live_df['pred_ridge'] = ridge_pipeline.predict(X_live)
    
    # Convert log returns to arithmetic returns
    live_df['pred_rf_arithmetic_equiv'] = np.expm1(live_df['pred_rf'])
    live_df['pred_ridge_arithmetic_equiv'] = np.expm1(live_df['pred_ridge'])
    
    # Rank cards cross-sectionally
    live_df['rf_rank'] = live_df['pred_rf'].rank(ascending=False, method='min').astype('Int64')
    live_df['ridge_rank'] = live_df['pred_ridge'].rank(ascending=False, method='min').astype('Int64')
    live_df['xwoba_rank'] = live_df['xwoba_t30'].rank(ascending=False, method='min').astype('Int64')    
    
    # Format and clean final output. 
    output_cols = [
        'snapshot_date', 
        'player_name', 
        'card_id', 
        'price_median_30d', 
        'sales_count_30d', 
        'pred_rf', 
        'pred_rf_arithmetic_equiv',
        'rf_rank', 
        'pred_ridge', 
        'pred_ridge_arithmetic_equiv',
        'ridge_rank', 
        'xwoba_t30', 
        'xwoba_rank'
    ]
    
    # Sort by the primary model's top rankings
    live_ranked = live_df[output_cols].sort_values('rf_rank', ascending=True).reset_index(drop=True)
    
    return live_ranked

if __name__ == "__main__":
    
    dataset_path = 'data/model_dataset.csv'
    output_dir = os.path.join('outputs', 'generated')
    output_path = os.path.join(output_dir, 'latest_predictions.csv')
    
    if os.path.exists(dataset_path):
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(dataset_path)
        
        # Check for missing columns
        required_structural = ['snapshot_date', 'future_end_date', 'target_28d_log_return', 'player_name', 'card_id']
        missing_structural = [c for c in required_structural if c not in df.columns]
        if missing_structural:
            raise ValueError(f"\nError! Model dataset missing required structural columns: {missing_structural}")
        
        # Check for missing features
        missing_features = [f for f in V3B_FEATURES if f not in df.columns]
        if missing_features:
            raise ValueError(f"\nError! Model dataset missing required features: {missing_features}")
            
        try:
            live_predictions_df = generate_live_predictions(df, V3B_FEATURES)
            live_predictions_df.to_csv(output_path, index=False)
            
            print(f"\nSuccess! Top 5 Random Forest Rankings for {live_predictions_df['snapshot_date'].iloc[0].strftime('%Y-%m-%d')}:")
            
            # Print a clean, cross-model comparison table
            display_cols = ['rf_rank', 'player_name', 'pred_rf_arithmetic_equiv', 'ridge_rank', 'xwoba_rank', 'price_median_30d']
            print(live_predictions_df[display_cols].head(5).to_string(index=False))
            
            print(f"\nSaved complete ranked predictions to '{output_path}'")
            
        except Exception as e:
            print(f"\nError! During prediction generation: {e}")
            sys.exit(1)
            
    else:
        print(f"\nError! '{dataset_path}' not found. Confirm that build_features.py has been run.")
        sys.exit(1)
