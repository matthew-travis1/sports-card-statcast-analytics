"""
Feature Engineering and Dataset Creation Module
---------------------------------------------------
Builds the core machine learning panel dataset using a strict Point-In-Time (PIT) 
walk-forward methodology. Prevents data leakage through calculation of a historical date T,
which restricts the model to data known on or before T. Calculates the forward 28-day returns
using only transaction prices strictly occurring in the (T, T+28] window. Uses the statistical
features from the 'V3B' feature candidate. Experimental V4/V5 market-response features
(momentum, velocity, card age) have been removed.
"""

import pandas as pd
import numpy as np
import os

def _compute_statcast_metrics(df_subset, prefix):
    """
    Calculates aggregated Statcast batting metrics for given time window.
    
    Parameters:
    -----------
    df_subset : pd.DataFrame
        Statcast events occurring strictly within the given historical time window.
    prefix : str
        String identifier (e.g.: 't30', 'long') added to output feature names.
        
    Returns:
    --------
    dict
        Dictionary of chosen and created Statcast features for the model.
    """
    # Return nulls if the player had no qualifying plate appearances in this time window
    if df_subset.empty or 'woba_denom' not in df_subset.columns or df_subset['woba_denom'].sum() == 0:
        empty_metrics = {f'{m}_{prefix}': np.nan for m in [
            'xwoba', 'woba', 'xwoba_gap', 'barrel_pct', 'hardhit_pct', 
            'k_pct', 'bb_pct', 'avg_ev', 'max_ev', 'sweet_spot_pct', 
            'xba', 'ba', 'xba_gap', 'xslg', 'slg', 'xslg_gap'
        ]}
        empty_metrics[f'pa_{prefix}'] = 0
        return empty_metrics
    
    # Plate Appearances: woba_denom accurately represents official PA.
    total_pa = df_subset['woba_denom'].sum()
    pa = int(total_pa)
    
    xwoba = df_subset['xwoba_event_value'].sum() / total_pa if 'xwoba_event_value' in df_subset.columns else np.nan
    woba = df_subset['woba_value'].sum() / total_pa if 'woba_value' in df_subset.columns else np.nan
    xwoba_gap = xwoba - woba if pd.notna(xwoba) and pd.notna(woba) else np.nan
    
    # Batted Ball Events: Isolate number of BBE for contact-specific metrics.
    bbe_df = df_subset[df_subset['launch_speed'].notna()]
    total_bbe = len(bbe_df)
    
    barrel_pct = (bbe_df['is_barrel'].sum() / total_bbe * 100) if total_bbe > 0 else np.nan
    hardhit_pct = (bbe_df['is_hard_hit'].sum() / total_bbe * 100) if total_bbe > 0 else np.nan
    avg_ev = bbe_df['launch_speed'].mean() if total_bbe > 0 else np.nan
    max_ev = bbe_df['launch_speed'].max() if total_bbe > 0 else np.nan
    
    ss_count = ((bbe_df['launch_angle'] >= 8) & (bbe_df['launch_angle'] <= 32)).sum()
    sweet_spot_pct = (ss_count / total_bbe * 100) if total_bbe > 0 else np.nan
    
    k_count = df_subset['events'].isin(['strikeout', 'strikeout_double_play']).sum()
    bb_count = df_subset['events'].isin(['walk', 'intent_walk']).sum()
    k_pct = (k_count / total_pa * 100)
    bb_pct = (bb_count / total_pa * 100)
    
    # At-Bats: Standard AB events. 
    ab_events_list = [
        'single', 'double', 'triple', 'home_run', 'strikeout', 'strikeout_double_play', 
        'field_out', 'grounded_into_dp', 'double_play', 'force_out', 'fielders_choice', 
        'fielders_choice_out', 'lineout', 'pop_out', 'flyout'
    ]
    hits = df_subset['events'].isin(['single', 'double', 'triple', 'home_run']).sum()
    at_bats = df_subset['events'].isin(ab_events_list).sum()
    
    tb = (df_subset['events'].eq('single').sum() * 1 + 
          df_subset['events'].eq('double').sum() * 2 + 
          df_subset['events'].eq('triple').sum() * 3 + 
          df_subset['events'].eq('home_run').sum() * 4)
          
    ba = hits / at_bats if at_bats > 0 else np.nan
    slg = tb / at_bats if at_bats > 0 else np.nan
    
    # xBA and xSLG calculated using speedangle estimates and BBE as denominator.
    xba = bbe_df['estimated_ba_using_speedangle'].mean() if 'estimated_ba_using_speedangle' in bbe_df.columns else np.nan
    xslg = bbe_df['estimated_slg_using_speedangle'].mean() if 'estimated_slg_using_speedangle' in bbe_df.columns else np.nan
    
    xba_gap = xba - ba if pd.notna(xba) and pd.notna(ba) else np.nan
    xslg_gap = xslg - slg if pd.notna(xslg) and pd.notna(slg) else np.nan
    
    return {
        f'xwoba_{prefix}': xwoba, f'woba_{prefix}': woba, f'xwoba_gap_{prefix}': xwoba_gap,
        f'barrel_pct_{prefix}': barrel_pct, f'hardhit_pct_{prefix}': hardhit_pct,
        f'k_pct_{prefix}': k_pct, f'bb_pct_{prefix}': bb_pct, f'avg_ev_{prefix}': avg_ev,
        f'max_ev_{prefix}': max_ev, f'sweet_spot_pct_{prefix}': sweet_spot_pct,
        f'xba_{prefix}': xba, f'ba_{prefix}': ba, f'xba_gap_{prefix}': xba_gap,
        f'xslg_{prefix}': xslg, f'slg_{prefix}': slg, f'xslg_gap_{prefix}': xslg_gap,
        f'pa_{prefix}': pa
    }

def build_point_in_time_panel(sales_df, statcast_df, analysis_start='2025-03-27', min_sales=3):
    """
    Constructs the V3B modeling dataset by walking forward through historical snapshots.
    
    Parameters:
    -----------
    sales_df : pd.DataFrame
        Cleaned and filtered historical sales data.
    statcast_df : pd.DataFrame
        Plate appearance level Statcast event logs.
    analysis_start : str
        ISO date string marking the beginning of the evaluation period.
    min_sales : int
        Minimum historical and future liquidity (number of sales) required to form a valid row.
        
    Returns:
    --------
    pd.DataFrame
        The final point-in-time, cross-sectional modeling dataset.
    """
    print(f"Building Weekly Research Panel With Min Sales = {min_sales}...")
    
    sales_df['sale_date'] = pd.to_datetime(sales_df['sale_date']).dt.tz_localize(None).dt.normalize()
    statcast_df['game_date'] = pd.to_datetime(statcast_df['game_date']).dt.tz_localize(None).dt.normalize()
    
    # Establish dynamic time boundary for the walk-forward loop
    data_cutoff_date = sales_df['sale_date'].max().normalize()
    analysis_start_ts = pd.Timestamp(analysis_start)
    snapshot_dates = pd.date_range(start=analysis_start_ts, end=data_cutoff_date, freq='W-SUN', normalize=True)
    
    # Restrict modeling primarily to the active MLB season windows (when batting data actually exists)
    snapshot_dates = snapshot_dates[snapshot_dates.month.isin([4, 5, 6, 7, 8, 9, 10])]
    
    cards = sales_df['card_id'].unique()
    panel_rows = []
    
    for card in cards:
        card_sales = sales_df[sales_df['card_id'] == card].sort_values('sale_date')
        player = card_sales['player_name'].iloc[0]
        player_sc = statcast_df[statcast_df['player_name'] == player].sort_values('game_date')
        
        for snap_date in snapshot_dates:
            # Step 1: Liquidity & Current Price (Trailing 30 Days)
            bkw_start = snap_date - pd.Timedelta(days=30)
            bkw_sales = card_sales[(card_sales['sale_date'] > bkw_start) & (card_sales['sale_date'] <= snap_date)]
            
            sales_count_30d = len(bkw_sales)
            if sales_count_30d >= min_sales:
                price_median_30d = bkw_sales['sold_price'].median()
                price_mean_30d = bkw_sales['sold_price'].mean()
                price_dispersion_30d = (bkw_sales['sold_price'].std() / price_mean_30d) if price_mean_30d > 0 else np.nan
            else:
                # If liquidity requirement fails, we skip feature calculation and drop the row downstream
                continue

            # Step 2: Target Date Calculation (Forward 28 Days)
            fwd_end = snap_date + pd.Timedelta(days=28)
            
            # Nullify target if the forward period hasn't finished yet
            if fwd_end > data_cutoff_date:
                future_28d_price, future_sales_count, target_28d_log_return = np.nan, 0, np.nan
            else:
                fwd_sales = card_sales[(card_sales['sale_date'] > snap_date) & (card_sales['sale_date'] <= fwd_end)]
                future_sales_count = len(fwd_sales)
                future_28d_price = fwd_sales['sold_price'].median() if future_sales_count >= min_sales else np.nan
                
                if pd.notna(price_median_30d) and pd.notna(future_28d_price) and price_median_30d > 0:
                    target_28d_log_return = np.log(future_28d_price / price_median_30d)
                else:
                    target_28d_log_return = np.nan

            # Step 3: Statcast Features (V3B Model)
            # ---------------------------------------------------------
            sc_long = player_sc[player_sc['game_date'] <= snap_date]
            sc_season = player_sc[(player_sc['game_date'].dt.year == snap_date.year) & (player_sc['game_date'] <= snap_date)]
            sc_t30 = player_sc[(player_sc['game_date'] > snap_date - pd.Timedelta(days=30)) & (player_sc['game_date'] <= snap_date)]
            
            metrics_t30 = _compute_statcast_metrics(sc_t30, 't30')
            metrics_season = _compute_statcast_metrics(sc_season, 'season')
            metrics_long = _compute_statcast_metrics(sc_long, 'long')
            
            # Require at least some baseline career data (greater than 0 PA) to create relative metrics
            if metrics_long['pa_long'] == 0:
                continue 
            
            # Create relative performance metrics ("How hot is he relative to his standard baseline?")
            xwoba_vs_long = metrics_t30['xwoba_t30'] - metrics_long['xwoba_long']
            barrel_vs_long = metrics_t30['barrel_pct_t30'] - metrics_long['barrel_pct_long']
            hardhit_vs_long = metrics_t30['hardhit_pct_t30'] - metrics_long['hardhit_pct_long']
            
            active_30d = int(metrics_t30['pa_t30'] >= 10)
            days_since_game = (snap_date - sc_long['game_date'].max()).days if not sc_long.empty else 999
            missing_statcast_t30 = int(metrics_t30['pa_t30'] == 0)

            # Assemble row (Strictly V3B components)
            row_data = {
                'snapshot_date': snap_date, 
                'future_end_date': fwd_end, 
                'player_name': player, 
                'card_id': card,
                'price_median_30d': price_median_30d, 
                'sales_count_30d': sales_count_30d, 
                'price_dispersion_30d': price_dispersion_30d,
                'future_28d_price': future_28d_price, 
                'future_sales_count': future_sales_count, 
                'target_28d_log_return': target_28d_log_return,
                'active_30d': active_30d, 
                'days_since_game': days_since_game,
                'xwoba_vs_long': xwoba_vs_long, 
                'barrel_vs_long': barrel_vs_long, 
                'hardhit_vs_long': hardhit_vs_long,
                'missing_statcast_t30': missing_statcast_t30
            }
            row_data.update(metrics_long)
            row_data.update(metrics_season)
            row_data.update(metrics_t30)
            panel_rows.append(row_data)

    final_df = pd.DataFrame(panel_rows)
    
    # Calculate cross-sectional Z-Scores for V3B features to evaluate relative rank
    z_features = ['xwoba_t30', 'barrel_pct_t30', 'hardhit_pct_t30']
    for feat in z_features:
        if feat in final_df.columns:
            final_df[f'z_{feat}'] = final_df.groupby('snapshot_date')[feat].transform(
                lambda x: (x - x.mean()) / x.std() if (len(x.dropna()) >= 10 and x.std() > 0) else np.nan
            )
            
    # Record mathematical extremes for possible errors
    if 'target_28d_log_return' in final_df.columns:
        extreme_targets = final_df[final_df['target_28d_log_return'].abs() > 0.50]
        if not extreme_targets.empty:
            extreme_target_file = 'data/extreme_targets.csv'
            extreme_targets.to_csv(extreme_target_file, index=False)
            
    return final_df

if __name__ == "__main__":
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    SALES_CSV = 'data/clean_historical_sales.csv'
    STATCAST_CSV = 'data/statcast_pa_log.csv'
    OUTPUT_CSV = 'data/model_dataset.csv'
    
    if os.path.exists(SALES_CSV) and os.path.exists(STATCAST_CSV):
        print("\nLoading historical datasets...")
        sales_df = pd.read_csv(SALES_CSV)
        statcast_df = pd.read_csv(STATCAST_CSV, low_memory=False)
        
        final_df = build_point_in_time_panel(sales_df, statcast_df, analysis_start='2025-03-27', min_sales=3)
        final_df.to_csv(OUTPUT_CSV, index=False)
        
        print(f"\nSuccess! Created ML features and saved to '{OUTPUT_CSV}'")
        print(f"Total modeling rows generated: {len(final_df)}")
    else:
        print("\nError! Missing files.")
        print(f"Confirm '{SALES_CSV}' and '{STATCAST_CSV}' exist before building features.")