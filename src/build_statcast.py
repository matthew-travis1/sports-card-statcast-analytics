"""
Statcast Data Collection Module
--------------------------------
Collects and processes raw MLB Statcast event data for target players.
Transforms pitch-by-pitch tracking metrics into clean, plate-appearance-level
(PA) data logs containing exit velocity, launch angle, xwOBA, barrels, and hard-hit metrics.
"""

import pandas as pd
import requests
import urllib.parse
import os
import sys
from datetime import datetime, timezone
from pybaseball import statcast_batter, cache

# Enable local caching to prevent redundant requests to Statcast servers
cache.enable()


def get_mlbam_id(player_name):
    """
    Retrieves the official MLBAM (Major League Baseball Advanced Media) ID for a player.
    Implements MLB's official API directly to handle special characters, accents, and suffixes.

    Parameters:
    -----------
    player_name : str
        First and last name of the player.

    Returns:
    --------
    int or None
        The numeric MLBAM ID if found, otherwise None.
        
    Notes:
    ------
    Limitation: Sorting by birthDate descending to pick the youngest active player 
    is a heuristic to duplicate player names (e.g.: father/son duos). This has 
    the potential risk of selecting the wrong player if multiple active players 
    share the exact same name.
    """
    # URL-encode the name to safely handle spaces, accents, and punctuation
    encoded_name = urllib.parse.quote(player_name.strip())
    url = f"https://statsapi.mlb.com/api/v1/people/search?names={encoded_name}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            people = data.get('people', [])
            
            if people:
                # Sort by birthDate descending so the active modern player 
                # is selected first in cases of shared exact names.
                people.sort(key=lambda x: x.get('birthDate', '1900-01-01'), reverse=True)
                mlbam_id = people[0]['id']
                return mlbam_id
                
    except requests.exceptions.RequestException as e:
        print(f"API Error fetching MLBAM ID for {player_name}: {e}")
        
    print(f"Error! Could not find MLBAM ID for {player_name}")
    return None


def fetch_statcast_terminal_pitches(mlbam_id, player_name, start_date="2024-03-01", end_date=None):
    """
    Fetches raw Statcast data and isolates terminal pitches (the final pitch of each PA).
    Collects core PA-level batting metrics including xwOBA, Hard-Hit status, and Barrel rates.

    Parameters:
    -----------
    mlbam_id : int
        The player's official MLBAM ID.
    player_name : str
        Player name for tagging the output DataFrame.
    start_date : str, default "2024-03-01"
        ISO formatted start date (YYYY-MM-DD) for historical window.
    end_date : str, optional
        ISO formatted end date. Defaults to current UTC date.

    Returns:
    --------
    pd.DataFrame
        Chronologically sorted DataFrame of terminal plate appearances.
    """
    if end_date is None:
        end_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        
    print(f"Fetching pitch data for {player_name} ({start_date} to {end_date})...")
    
    try:
        df = statcast_batter(start_dt=start_date, end_dt=end_date, player_id=mlbam_id)
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"Error! Failed to fetch statcast data. {e}")
        return pd.DataFrame()
        
    if df.empty:
        return pd.DataFrame()

    # Keep only events explicitly included in the wOBA denominator (check for woba_denom == 1.0).
    # This filters out non-qualifying events (e.g.: caught stealing, pickoffs) in order 
    # to isolate the final, qualifying event of each Plate Appearance.
    terminal_pitches = df.dropna(subset=['events']).copy()
    terminal_pitches = terminal_pitches[terminal_pitches['woba_denom'] == 1.0].copy()
    
    # Standardize time zone representation for seamless datetime operations
    terminal_pitches['game_date'] = pd.to_datetime(terminal_pitches['game_date']).dt.tz_localize(None)
    
    # Create a unified expected outcome value (xwoba_event_value):
    # Combines expected wOBA for batted balls (derived from launch angle and exit velocity) 
    # with actual wOBA values for non-batted-ball PAs (walks, strikeouts, HBP) where 
    # launch angle and exit velocity do not exist.
    terminal_pitches['xwoba_event_value'] = terminal_pitches['estimated_woba_using_speedangle'].fillna(
        terminal_pitches['woba_value']
    )
    
    # Hard-hit contact classification:
    # Exit Velocity >= 95 mph is the official MLB threshold for a Hard-Hit ball.
    # Non-batted balls (NaN launch_speed) must return NaN, not 0, to ensure 
    # hard-hit percentages correctly use only Batted Ball Events as the denominator, not total PAs.
    terminal_pitches['is_hard_hit'] = terminal_pitches['launch_speed'].apply(
        lambda x: 1.0 if pd.notna(x) and x >= 95.0 else (0.0 if pd.notna(x) else float('nan'))
    )
    
    # Barrel classification:
    # The class code 6.0 represents a 'Barrel' in Statcast's launch_speed_angle model.
    # Like hard-hit balls, non-contact events must resolve to NaN for barrel percent calculations.
    terminal_pitches['is_barrel'] = terminal_pitches['launch_speed_angle'].apply(
        lambda x: 1.0 if x == 6.0 else (0.0 if pd.notna(x) else float('nan'))
    )
    
    keep_cols = [
        'game_date', 'events', 'pitch_type', 'release_speed', 'launch_speed', 
        'launch_angle', 'is_hard_hit', 'is_barrel', 'woba_value', 'woba_denom', 
        'xwoba_event_value', 'estimated_ba_using_speedangle', 'estimated_slg_using_speedangle'
    ]
    
    available_cols = [c for c in keep_cols if c in terminal_pitches.columns]
    
    clean_df = terminal_pitches[available_cols].copy()
    clean_df.insert(0, 'player_name', player_name)
    
    # Ensure strict chronological ordering for future point-in-time modeling
    return clean_df.sort_values(by='game_date').reset_index(drop=True)


if __name__ == "__main__":
    
    os.makedirs('data', exist_ok=True)
    csv_path = 'data/statcast_pa_log.csv'
    
    try:
        card_data = pd.read_csv('data/card_names.csv')
    except FileNotFoundError:
        print("\nError! No 'data/card_names.csv' file found.")
        print("Please ensure the target universe file exists before running.")
        sys.exit(1)
        
    unique_players = card_data['player_name'].unique()
    all_player_logs = []
    
    for player in unique_players:
        mlbam_id = get_mlbam_id(player)
        if mlbam_id:
            player_log = fetch_statcast_terminal_pitches(mlbam_id, player, start_date="2024-03-01")
            if not player_log.empty:
                all_player_logs.append(player_log)
                
    if all_player_logs:
        master_statcast_log = pd.concat(all_player_logs, ignore_index=True)
        master_statcast_log.to_csv(csv_path, index=False)
        print(f"\nSuccess! Saved Statcast Plate Appearance Log to '{csv_path}'")
    else:
        print("\nWarning! No Statcast records were collected for any players.")
