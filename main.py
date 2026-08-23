"""
Sports Card Statcast Analytics Main Pipeline
------------------------------------------------------
This is the entry point for the repository. It sequentially runs the 
machine learning and backtesting pipeline using the prepared datasets.

IMPORTANT! Data Collection and Build Statcast (Modules 1 & 2) require API calls 
and scraping. They are bypassed by default. If you wish to run them,
execute the modules directly from the src/ folder.
"""

import sys
import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_pipeline():
    print("Sports Card Statcast Analytics Main Pipeline")
    
    # Check for required input data for Module 3
    sales_path = os.path.join(BASE_DIR, 'data', 'clean_historical_sales.csv')
    statcast_path = os.path.join(BASE_DIR, 'data', 'statcast_pa_log.csv')
    
    missing_files = []
    if not os.path.exists(sales_path):
        missing_files.append(sales_path)
    if not os.path.exists(statcast_path):
        missing_files.append(statcast_path)
        
    if missing_files:
        print(f"\nError! Required prepared datasets not found: {missing_files}")
        print("Please ensure the required prepared datasets are available.")
        print("You can run Steps 1 & 2 manually if you have API credentials.")
        sys.exit(1)

    print("\nStep 1 & 2: Data Collection + Cleaning & Build Statcast...")
    print("Bypassed. (Requires API scraper execution. See src/ modules to run manually).")
    
    try:
        print("\nStep 3: Build Features...")
        print("Running: src/build_features.py")
        subprocess.run([sys.executable, os.path.join("src", "build_features.py")], check=True, cwd=BASE_DIR)
        
        print("\nStep 4: Train Models...")
        print("Running: src/train_models.py")
        subprocess.run([sys.executable, os.path.join("src", "train_models.py")], check=True, cwd=BASE_DIR)
        
        print("\nStep 5: Cross-Sectional Backtest...")
        print("Running: src/backtest.py")
        subprocess.run([sys.executable, os.path.join("src", "backtest.py")], check=True, cwd=BASE_DIR)
        
        print("\nStep 6: Generate Latest Predictions...")
        print("Running: src/generate_predictions.py")
        subprocess.run([sys.executable, os.path.join("src", "generate_predictions.py")], check=True, cwd=BASE_DIR)
        
    except subprocess.CalledProcessError:
        print(f"\nError! Pipeline failed during execution of the module.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError! Unexpected failure: {e}")
        sys.exit(1)

    print("\nFull Pipeline Complete!")

if __name__ == "__main__":
    
    run_pipeline()
