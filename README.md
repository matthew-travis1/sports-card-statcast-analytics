# Sports Card Statcast Analytics

## Project Overview

Sports Card Statcast Analytics is a Python data science project that combines MLB Statcast performance data with historical sports card transactions to investigate whether advanced batting metrics can help predict a player's relative card price performance over the following 28 days.
The project is designed as a **point-in-time forecasting experiment**. Every model uses only information that would have been available on the historical prediction date, helping prevent future-data leakage.
The primary goal is to evaluate whether eligible sports cards can be **ranked by their expected relative performance** over the following 28 days.

## Demo

A condensed public dataset is included in `data/sample/` so the
machine learning pipeline can be run without external API calls.

Example outputs generated from the public sample are available in
`outputs/example_results/`, including:

- `sample_backtest_summary.csv` — model comparison and backtest metrics
- `sample_latest_predictions.csv` — latest cross-sectional card rankings

The public sample is a reduced demonstration dataset and should not be
interpreted as reproducing results from the complete research dataset.

Note: Data Visualization in progress...

## Tech Stack

* Python
* pandas
* NumPy
* scikit-learn
* SciPy
* pybaseball / MLB Statcast
* requests
* Apify API client
* Git
* GitHub

## How to Run Locally

Clone the repository and install the project dependencies:

bash
git clone https://github.com/matthew-travis1/sports-card-statcast-analytics.git
cd sports-card-statcast-analytics
python -m pip install -r requirements.txt

After preparing the sample data as described in the Demo section, run:
bash
python main.py

## What I Learned

This project has provided practical experience with:

* Combining sports analytics and transaction datasets
* Building point-in-time machine learning features
* Preventing future-data leakage in predictive modeling
* Creating forward-looking return targets from transaction data
* Using walk-forward validation instead of random train/test splitting
* Evaluating machine learning models as ranking systems
* Comparing linear and nonlinear models

## Known Limitations

* The historical sports card transaction sample is still relatively small.
* Some cards and players have limited transaction liquidity.
* Overlapping 28-day prediction periods mean observations should not be treated as completely independent.
* Current research results are preliminary and should not be interpreted as evidence of a proven trading strategy.
* Portfolio returns are reported before transaction costs and other trading frictions.
* Results may be sensitive to a small number of unusually strong or weak card returns.

## Next Steps

Future research priorities and planned repository updates include:

*   **Data Visualization:** Developing clean, presentation-ready charts to highlight model comparisons and portfolio returns.
*   **Robustness Diagnostics:** Expanding the backtest suite to stress-test the model against outliers, random noise, and realistic portfolio turnover.
*   **Liquidity Analysis:** Investigating how different levels of market liquidity impact the reliability of the ranking signals.
*   **Statistical Analysis:** Expand statistical testing to better evaluate the predictive power and significance of individual baseball metrics.
*   **Dataset Expansion:** Collecting additional out-of-sample transaction data to ensure model stability across a larger, more diverse card universe.

## Development Assistance

This project was developed with assistance from AI tools, including ChatGPT and Gemini, for code review, debugging, documentation, and implementation guidance. All research design decisions, validation logic, model evaluation, and final code review were performed and verified by the project author.

