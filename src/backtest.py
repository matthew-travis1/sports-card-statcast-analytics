"""
Backtest and Statistical Evaluation Module
---------------------------------------------
Evaluates machine learning predictions using cross-sectional portfolio sorting.
Portfolio results are reported as gross arithmetic returns for economic
interpretation, before transaction costs and other trading frictions,
though models train on log returns for mathematical symmetry. Forces a common universe
comparison across the Random Forest, Ridge baseline, and xwOBA benchmark to prevent
biased evaluation.
"""

import pandas as pd
import numpy as np
import os
import sys
from scipy.stats import spearmanr

MIN_CARDS_FOR_PORTFOLIO = 5
MIN_CARDS_FOR_SPEARMAN = 10

def evaluate_portfolio_snapshot(group, score_col, pct=0.20):
    """
    Evaluates a single weekly cross-section by ranking predictions into portfolios.
    
    Parameters:
    -----------
    group : pd.DataFrame
        A single week's cross-sectional dataset.
    score_col : str
        The column name of the model predictions to sort by.
    pct : float, default 0.20
        The fractional size of the top/bottom portfolios (e.g.: Top/Bottom 20%).
        
    Returns:
    --------
    dict
        Dictionary containing the arithmetic portfolio returns, market returns, 
        and total card count.
    """
    sorted_group = group.sort_values(score_col, ascending=False).reset_index(drop=True)
    n_total = len(sorted_group)
    n_top = max(1, int(np.ceil(n_total * pct)))
    
    # Evaluates in arithmetic terms (np.expm1)
    top_20_arithmetic = np.expm1(sorted_group.head(n_top)['target_28d_log_return']).mean()
    bot_20_arithmetic = np.expm1(sorted_group.tail(n_top)['target_28d_log_return']).mean()
    mkt_arithmetic = np.expm1(sorted_group['target_28d_log_return']).mean()
    
    return {
        'top_20_arith': top_20_arithmetic, 
        'bot_20_arith': bot_20_arithmetic, 
        'mkt_arith': mkt_arithmetic,
        'n_cards': n_total
    }

def evaluate_dataset_subset(df_subset, model_col):
    """
    Iterates through subset of data week-by-week to calculate portfolio statistics 
    and Spearman coefficients.
    
    Parameters:
    -----------
    df_subset : pd.DataFrame
        The cross-sectional dataset to evaluate over time.
    model_col : str
        The column name of the model predictions to sort by.
        
    Returns:
    --------
    tuple(pd.DataFrame, list)
        A DataFrame of weekly portfolio returns and a list of valid weekly 
        Spearman coefficients.
    """
    weekly_stats = []
    weekly_corrs = []
    
    for _, group in df_subset.groupby('snapshot_date'):
        if len(group) >= MIN_CARDS_FOR_PORTFOLIO:
            stats = evaluate_portfolio_snapshot(group, model_col, pct=0.20)
            weekly_stats.append(stats)
            
        if len(group) >= MIN_CARDS_FOR_SPEARMAN:
            corr, _ = spearmanr(group['target_28d_log_return'], group[model_col])
            if not np.isnan(corr):
                weekly_corrs.append(corr)
                
    stats_df = pd.DataFrame(weekly_stats) if weekly_stats else pd.DataFrame()
    return stats_df, weekly_corrs

def print_exclusion_log(df):
    """
    Logs rows dropped due to missing predictions regarding the Common Comparison Universe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The complete dataset containing targets and model predictions.
        
    Returns:
    --------
    None
        Prints the exclusion log to the console.
    """
    print("\nWeekly Exclusion and Eligibility Log:")
    print(f"{'Date':<12} | {'Total Rows':<10} | {'Valid RF':<10} | {'Valid Ridge':<11} | {'Valid xwOBA':<12} | {'Spearman Elig.':<15} | {'Exclusion Reason'}")
    
    required_cols = [
        'target_28d_log_return',
        'pred_rf',
        'pred_ridge',
        'xwoba_t30'
    ]
    
    for snap_date, group in df.groupby('snapshot_date'):
        total_rows = len(group)
        valid_rf = group['pred_rf'].notna().sum() if 'pred_rf' in group.columns else 0
        valid_ridge = group['pred_ridge'].notna().sum() if 'pred_ridge' in group.columns else 0
        valid_xwoba = group['xwoba_t30'].notna().sum() if 'xwoba_t30' in group.columns else 0
        
        # Determine how many rows have targets and predictions for all models
        common_universe = len(group.dropna(subset=required_cols))
        
        spearman_elig = "Yes" if common_universe >= MIN_CARDS_FOR_SPEARMAN else "No"
        
        reason = "None"
        if common_universe < MIN_CARDS_FOR_SPEARMAN:
            if common_universe == 0:
                reason = "No target/prediction data"
            else:
                reason = f"Cross-section too small (<{MIN_CARDS_FOR_SPEARMAN})"
                
        print(f"{snap_date.strftime('%Y-%m-%d'):<12} | {total_rows:<10} | {valid_rf:<10} | {valid_ridge:<11} | {valid_xwoba:<12} | {spearman_elig:<15} | {reason}")

def run_backtest_suite(preds_df):
    """
    Runs the complete cross-sectional evaluation pipeline and returns structured data.
    
    Parameters:
    -----------
    preds_df : pd.DataFrame
        The raw predictions dataset output by the model training module.
        
    Returns:
    --------
    pd.DataFrame
        A summary DataFrame containing the backtest results for each model and benchmark.
    """
    print_exclusion_log(preds_df)
    print("\nBacktest Suite: Common Comparison Universe (Random Forest vs Ridge vs xwOBA)")
    
    models_to_test = ['pred_rf', 'pred_ridge', 'xwoba_t30']
    
    # Confirm exact same rows across all models for identical market benchmark testing
    eval_df = preds_df.dropna(subset=['target_28d_log_return'] + models_to_test).copy()
    
    summary_results = []
    
    for model_col in models_to_test:
        stats_df, weekly_corrs = evaluate_dataset_subset(eval_df, model_col)
                
        if stats_df.empty:
            continue
            
        avg_top_20_arith = stats_df['top_20_arith'].mean() * 100
        avg_bot_20_arith = stats_df['bot_20_arith'].mean() * 100
        avg_mkt_arith = stats_df['mkt_arith'].mean() * 100
        
        spread_20 = avg_top_20_arith - avg_bot_20_arith
        excess_20 = avg_top_20_arith - avg_mkt_arith
        excess_20_series = stats_df['top_20_arith'] - stats_df['mkt_arith']
        pct_beating_mkt = (excess_20_series > 0).mean() * 100
        
        actual_mean_spearman = np.mean(weekly_corrs) if weekly_corrs else np.nan
        
        summary_results.append({
            'model': model_col,
            'mean_spearman': actual_mean_spearman,
            'top_20_return': avg_top_20_arith,
            'bottom_20_return': avg_bot_20_arith,
            'top_bottom_spread': spread_20,
            'market_return': avg_mkt_arith,
            'excess_vs_market': excess_20,
            'pct_weeks_beating_market': pct_beating_mkt,
            'n_weeks': len(stats_df),
            'n_spearman_weeks': len(weekly_corrs),
            'avg_cards_per_week': stats_df['n_cards'].mean()
        })
        
        print(f"\nEvaluation Results: [{model_col.upper()}] (Weeks: {len(stats_df)} | Avg Cards/Wk: {stats_df['n_cards'].mean():.1f})")
        print(f"Mean Weekly Spearman    : {actual_mean_spearman:.4f}")
        print(f"Top 20% Gross Return    : {avg_top_20_arith:+.2f}%")
        print(f"Bot 20% Gross Return    : {avg_bot_20_arith:+.2f}%")
        print(f"Market Gross Benchmark  : {avg_mkt_arith:+.2f}%")
        print(f"Top 20% - Bot 20% Spread: {spread_20:+.2f}%")
        print(f"Top 20% Excess vs Mkt   : {excess_20:+.2f}%")
        print(f"% Weeks Beating Market  : {pct_beating_mkt:.1f}%")
                
    return pd.DataFrame(summary_results)

if __name__ == "__main__":
    
    dataset_path = 'data/ml_predictions.csv'
    output_dir = os.path.join('outputs', 'generated')
    
    if os.path.exists(dataset_path):
        os.makedirs(output_dir, exist_ok=True)
        preds = pd.read_csv(dataset_path, parse_dates=['snapshot_date', 'future_end_date'])
        
        summary_df = run_backtest_suite(preds)
        
        if not summary_df.empty:
            summary_path = os.path.join(output_dir, 'backtest_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"\nSuccess! Saved backtest results to '{summary_path}'")
        else:
            print(f"\nError! Backtest completed but produced no summary results.")
            sys.exit(1)
    else:
        print(f"Error! '{dataset_path}' not found. Run train_models.py first.")
        sys.exit(1)
