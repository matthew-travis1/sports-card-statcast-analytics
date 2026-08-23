# Data Directory

Full transaction and Statcast datasets are excluded from this repository.
To allow reviewers to test the machine learning pipeline locally without external API calls,
a condensed sample dataset is provided.

## Sample Data (`data/sample/`)

This subset focuses on 50 highly traded players to maintain sufficient data density for the cross-sectional ranking models.

* **`sample_clean_sales.csv`**: Historical card transactions used for price and liquidity features.
* **`sample_statcast_pa_log.csv`**: Statcast events used to construct point-in-time performance features.
* **`sample_card_names.csv`**: Card-to-player mapping used by the data-collection module.

## How to Run the Demo

The orchestrator (`main.py`) expects the prepared files to be in the root `data/` directory.
To run the pipeline using the sample data:

1. Copy the required sample files into the root `data/` directory.
2. Remove the `sample_` prefix:
   * `sample_clean_historical_sales.csv` ➔ `clean_historical_sales.csv`
   * `sample_statcast_pa_log.csv` ➔ `statcast_pa_log.csv`
3. Run the pipeline from the repository root:
   ```bash
   python main.py
   
Note: The public sample is a reduced demonstration dataset and should not be interpreted as reproducing the results from the complete research dataset.