"""
Data Collection and Cleaning Module
-------------------------------------
Collects and cleans historical sports card transactions from multiple APIs.
This module is responsible for ensuring that we only model using data 
from the base Topps Chrome PSA 10 card sales rather than a combination of other 
parallels, autographs, and ungraded variations.
"""

import pandas as pd
import os
import time
import requests
import re
from apify_client import ApifyClient

def _clean_and_filter_transactions(raw_df, player, card_type, card_num, source):
    """
    Helper function to filter transactions and assign rejection reasons.
    
    Parameters:
    -----------
    raw_df : pd.DataFrame
        Raw transaction data from the API.
    player : str
        Target player's name.
    card_type : str
        Target card type (e.g., "Topps Chrome").
    card_num : str or int
        Target card number.
    source : str
        Source of the data ('parsebot' or 'apify') to handle column differences.
        
    Returns:
    --------
    tuple(pd.DataFrame, pd.DataFrame)
        Cleaned transaction data and separate data for rejected transactions.
    """
    df = raw_df.copy()
    df['rejection_reason'] = None
    
    if 'title' not in df.columns:
        df['rejection_reason'] = "Missing title"
        return pd.DataFrame(), df
        
    title_lower = df['title'].str.lower()
    
    # Require accepted listings to explicitly match Topps Chrome.
    mask_tc = title_lower.str.contains(r'\btopps chrome\b', regex=True)
    df.loc[~mask_tc & df['rejection_reason'].isna(), 'rejection_reason'] = "Missing 'Topps Chrome'"
    
    # Require accepted listings to explicitly match PSA 10.
    mask_psa = title_lower.str.contains(r'\bpsa\s*10\b', regex=True)
    df.loc[~mask_psa & df['rejection_reason'].isna(), 'rejection_reason'] = "Missing 'PSA 10'"
    
    # Remove Best Offers for Apify (API price is the initial listing price, which inflates sale prices).
    if source == 'apify' and 'isBestOfferAccepted' in df.columns:
        mask_bo = df['isBestOfferAccepted'].fillna(False).astype(bool)
        df.loc[mask_bo & df['rejection_reason'].isna(), 'rejection_reason'] = "Best Offer accepted"

    # Filter #1: Confirm that it is the intended base card.
    # Filter out parallels, autographs, and ungraded variations 
    # since they have different price distributions and volume.
    bad_words = [
        'lot', 'lots', 'read', 'refractors', 'xfractors', 'x-fractors', 'refractor', 'xfractor', 
        'x-fractor', 'update', 'mega box', 'megabox', 'blue', 'gold', 'green', 'orange', 'purple', 
        'black', 'sapphire', 'wave', 'mojo', 'sparkle', 'shimmer', 'speckle', 'magenta', 'promo', 
        'raywave', 'variations', 'asgc', 'sonar', 'logofractor', 'aqua', 'lava', 'sepia', 'pink', 
        'negative', 'allen', 'ginter', 'cosmic', 'variation', 'printing', 'prism', 'finest', 
        'pristine', 'platinum', 'fractor', 'sp', 'ssp', 'ben baller', 'sonic', 'lids', 'auto', 
        'autos', 'signed', 'autograph', 'autographed', 'raw', 'mba', 'sgc', 'bgs', 'cgc', 'psa 9', 'psa 8'
    ] 
    
    # Fix the autograph filter so "Non Auto" is not rejected just because it contains "auto"
    title_for_badwords = title_lower.str.replace(r'\bnon auto\b', 'non-auto-exempt', regex=True)
    bad_pattern = r'\b(?:' + '|'.join(map(re.escape, bad_words)) + r')\b'
    mask_bad = title_for_badwords.str.contains(bad_pattern, regex=True)
    df.loc[mask_bad & df['rejection_reason'].isna(), 'rejection_reason'] = "Matched invalid keyword"
    
    # Filter #2: Exact card number matching.
    # Filtering only the player name and year is insufficient because players 
    # often have multiple cards each year sharing the same general name.
    if pd.notna(card_num):
        clean_num = str(int(card_num)) if isinstance(card_num, (int, float)) else str(card_num).strip()
        escaped_num = re.escape(clean_num)
        good_pattern = rf'\b#?{escaped_num}\b'
        mask_num = title_lower.str.contains(good_pattern, regex=True)
        df.loc[~mask_num & df['rejection_reason'].isna(), 'rejection_reason'] = f"Missing card number ({clean_num})"
    
    # Filter #3: Remove price outliers via 1.5 * IQR.
    # Automatically drops extreme outliers caused by shill bidding, 
    # unpaid items, or mislabeled bulk sales without hardcoding limits.
    price_col = 'totalPrice' if source == 'apify' else 'price'
    if price_col in df.columns:
        df['target_price'] = pd.to_numeric(df[price_col], errors='coerce')
        df.loc[df['target_price'].isna() & df['rejection_reason'].isna(), 'rejection_reason'] = "Invalid price format"
        
        valid_price_mask = df['rejection_reason'].isna()
        if valid_price_mask.sum() > 0:
            q1 = df.loc[valid_price_mask, 'target_price'].quantile(0.25)
            q3 = df.loc[valid_price_mask, 'target_price'].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + (1.5 * iqr)
            lower_bound = q1 - (1.5 * iqr)
            
            mask_outlier = (df['target_price'] > upper_bound) | (df['target_price'] < lower_bound)
            df.loc[mask_outlier & valid_price_mask, 'rejection_reason'] = "Price outlier (1.5 IQR)"
    else:
        df.loc[df['rejection_reason'].isna(), 'rejection_reason'] = "Missing price column"
    
    # Filter #4: Normalize sale dates.
    # Standardizing datetime format is critical for the point-in-time 
    # walk-forward validation in the future modeling phase.
    date_col = 'endedAt' if source == 'apify' else 'date'
    if date_col in df.columns:
        df['clean_date_str'] = df[date_col].astype(str).str.replace(r'\s+[A-Z]{3,4}$', '', regex=True)
        df['sale_date'] = pd.to_datetime(df['clean_date_str'], errors='coerce', utc=True)
        df.loc[df['sale_date'].isna() & df['rejection_reason'].isna(), 'rejection_reason'] = "Invalid date format"
    else:
        df.loc[df['rejection_reason'].isna(), 'rejection_reason'] = "Missing date column"

    # Keep rejected listings instead of simply dropping them.
    # This creates an auditing system for reviewing any filtering errors.
    rejected_df = df[df['rejection_reason'].notna()].copy()
    rejected_df['player_name'] = player
    
    clean_df = df[df['rejection_reason'].isna()].copy()
    
    if not clean_df.empty:
        clean_df['player_name'] = player
        clean_df['card_id'] = f"{player}_{card_type}"
        clean_df['sold_price'] = clean_df['target_price']
        clean_df['is_best_offer'] = False
        
        if source == 'parsebot':
            clean_df['transaction_id'] = clean_df['id'].astype(str) if 'id' in clean_df.columns else clean_df.index.astype(str) + "_pb"
            clean_df['platform'] = clean_df['sold_via'] if 'sold_via' in clean_df.columns else 'Unknown'
        else:
            clean_df['transaction_id'] = clean_df['itemId'].astype(str) if 'itemId' in clean_df.columns else clean_df.index.astype(str) + "_ap"
            clean_df['platform'] = 'eBay'

        standard_cols = ['transaction_id', 'player_name', 'card_id', 'sale_date', 'sold_price', 'platform', 'is_best_offer', 'title']
        clean_df = clean_df[standard_cols].copy()
        
    return clean_df, rejected_df

def fetch_parsebot_historical_sales(card_data_df, api_key):
    """
    Fetches historical transaction data beyond eBay's native 90-day limit 
    using the Parsebot 130point API.
    
    Parameters:
    -----------
    card_data_df : pd.DataFrame
        Target universe of players and their specific base card information  (year, card number).
    api_key : str
        API key for Parse.bot authentication.
        
    Returns:
    --------
    tuple(pd.DataFrame, pd.DataFrame)
        Cleaned transaction data and separate data for rejected transactions.
    """
    print("\nInitiating Parsebot Scraper...")
    all_clean_sales = []
    all_rejected_sales = []
    
    parse_url = "https://api.parse.bot/scraper/28d873f5-47d5-4c01-a275-e80c6b3fc610/search_sold_items"
    headers = {
        "X-API-Key": api_key,
        "API-Snapshot-Version": "5"
    }

    for _, row in card_data_df.iterrows():
        player = row['player_name']
        card_type = row['card_name']
        card_num = row['card_number']
        
        search_term = f"{player} {card_type}"
        print(f"\nParse.bot searching for \"{search_term}\"")

        params = {
            "limit": "1000",
            "sort": "EndTimeSoonest",
            "query": search_term,
            "marketplace": "all"
        }
        
        try:
            response = requests.get(parse_url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            
            raw_data = response.json()
            items_list = raw_data.get('data', {}).get('items', [])
            
            if items_list and len(items_list) > 0:
                raw_df = pd.json_normalize(items_list)
                clean_df, rejected_df = _clean_and_filter_transactions(raw_df, player, card_type, card_num, source='parsebot')
                
                if not rejected_df.empty:
                    all_rejected_sales.append(rejected_df)
                if not clean_df.empty:
                    all_clean_sales.append(clean_df)
                    print(f"Compiled {len(clean_df)} clean sales, ({len(rejected_df)} rejected).")
                    
                    pd.concat(all_clean_sales, ignore_index=True).to_csv('data/parsebot_checkpoint.csv', index=False)
                        
        except requests.exceptions.RequestException as e:
            print(f"Network/API Error fetching Parsebot data for {player}: {e}")
            
        # Pause due to third-party API rate limiting
        time.sleep(13) 

    clean_out = pd.concat(all_clean_sales, ignore_index=True) if all_clean_sales else pd.DataFrame()
    reject_out = pd.concat(all_rejected_sales, ignore_index=True) if all_rejected_sales else pd.DataFrame()
    return clean_out, reject_out

def fetch_apify_recent_sales(card_data_df, apify_token):
    """
    Fetches recent (<90 days) transaction data directly from eBay using Apify API.
    
    Parameters:
    -----------
    card_data_df : pd.DataFrame
        Target universe of players and their specific base card information  (year, card number).
    apify_token : str
        API key for Apify authentication.
        
    Returns:
    --------
    tuple(pd.DataFrame, pd.DataFrame)
        Cleaned transaction data and separate data for rejected transactions.
    """
    print("\nInitiating Apify Scraper...")
    client = ApifyClient(apify_token)
    all_clean_sales = []
    all_rejected_sales = []

    for _, row in card_data_df.iterrows():
        player = row['player_name']
        card_type = row['card_name']
        card_num = row['card_number']
        
        search_term = f"{player} {card_type}"
        print(f"\nApify searching for \"{search_term}\"")
        
        run_input = {
            "keywords": [search_term],
            "daysToScrape": 90,
            "count": 500,
            "ebaySite": "ebay.com",
            "sortOrder": "endedRecently",
            "itemLocation": "default",
            "itemCondition": "any",
            "includeCompletedListings": True,
        }
        
        try:
            run = client.actor("oTtB3VgfuE9GtxQt2").call(run_input=run_input)
            items_list = list(client.dataset(run.default_dataset_id).iterate_items())
            
            if items_list and len(items_list) > 0:
                raw_df = pd.DataFrame(items_list)
                clean_df, rejected_df = _clean_and_filter_transactions(raw_df, player, card_type, card_num, source='apify')
                
                if not rejected_df.empty:
                    all_rejected_sales.append(rejected_df)
                if not clean_df.empty:
                    all_clean_sales.append(clean_df)
                    print(f"Compiled {len(clean_df)} clean sales, ({len(rejected_df)} rejected).")
                    
                    pd.concat(all_clean_sales, ignore_index=True).to_csv('data/apify_checkpoint.csv', index=False)
                        
        except Exception as e:
            print(f"Error fetching Apify data for {player}: {e}")
            
    clean_out = pd.concat(all_clean_sales, ignore_index=True) if all_clean_sales else pd.DataFrame()
    reject_out = pd.concat(all_rejected_sales, ignore_index=True) if all_rejected_sales else pd.DataFrame()
    return clean_out, reject_out

if __name__ == "__main__":
    
    os.makedirs('data', exist_ok=True)
    
    csv_path = 'data/clean_historical_sales.csv'
    rejected_path = 'data/rejected_card_sales.csv'
    
    # Bypass API calls if the clean dataset already exists
    if os.path.exists(csv_path):
        print(f"\nChecking for locally stored data...")
        print(f"Found '{csv_path}'. Skipping API requests.")
        print(f"To run fresh APIs, delete '{csv_path}' from the data folder first.")
        
    else:
        try:
            card_data = pd.read_csv('data/card_names.csv')
        except FileNotFoundError:
            print(f"\nNo 'data/card_names.csv' file found. Please ensure the target universe file exists.")
            exit(1)
            
        pb_key = os.getenv("PARSEBOT_API_KEY")
        apify_key = os.getenv("APIFY_API_TOKEN")
        
        if not pb_key or not apify_key:
            raise ValueError("Missing API credentials. Please set PARSEBOT_API_KEY and APIFY_API_TOKEN environment variables.")
            
        print(f"\nFiring API scrapers for data collection and cleaning...")
        
        try:
            parsebot_df, pb_rejects = fetch_parsebot_historical_sales(card_data, pb_key)
        except Exception as e:
            print(f"\nERROR! Parse.bot halted: {e}")
            parsebot_df = pd.read_csv('data/parsebot_checkpoint.csv') if os.path.exists('data/parsebot_checkpoint.csv') else pd.DataFrame()
            pb_rejects = pd.DataFrame()

        try:
            apify_df, apify_rejects = fetch_apify_recent_sales(card_data, apify_key)
        except Exception as e:
            print(f"\nERROR! Apify halted: {e}")
            apify_df = pd.read_csv('data/apify_checkpoint.csv') if os.path.exists('data/apify_checkpoint.csv') else pd.DataFrame()
            apify_rejects = pd.DataFrame()

        print("\nMerging and saving clean datasets")
        combined_clean = pd.concat([parsebot_df, apify_df], ignore_index=True)
        
        if not combined_clean.empty:
            # Remove duplicates. Since transaction IDs are different across scrapers we
            # remove duplicated by player name, exact time, exact price, and exact title to prevent double-counting.
            combined_clean = combined_clean.drop_duplicates(subset=['player_name', 'sale_date', 'sold_price', 'title'], keep='last')
            combined_clean.to_csv(csv_path, index=False)
            print(f"Successfully saved {len(combined_clean)} total clean transactions to '{csv_path}'.")
        else:
            print("Error! No clean data collected from either API.")
            
        combined_rejects = pd.concat([pb_rejects, apify_rejects], ignore_index=True)
        if not combined_rejects.empty:
            combined_rejects.to_csv(rejected_path, index=False)
            print(f"Successfully saved {len(combined_rejects)} rejected transactions to '{rejected_path}'.")