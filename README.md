# Sports Card Statcast Analytics

## Project Overview

Sports Card Statcast Analytics is a Python data science project that combines MLB Statcast performance data with historical sports card transactions to investigate whether advanced batting metrics can help predict a player's relative card price performance over the following 28 days.

The project is designed as a **point-in-time forecasting experiment**. Every model uses only information that would have been available on the historical prediction date, helping prevent future-data leakage.

The primary goal is not to predict an exact future card price. Instead, the project evaluates whether eligible sports cards can be **ranked by their expected relative performance** over the following 28 days.

## Demo

A project workflow diagram and example model results will be added as the GitHub version of the pipeline is completed.

## Tech Stack

* Python
* pandas
* NumPy
* scikit-learn
* statsmodels
* pybaseball / MLB Statcast
* Git
* GitHub

## How to Run Locally

The repository is currently being converted from a research project into a reproducible GitHub portfolio project.

Final setup and execution instructions will be added once the core pipeline has been migrated and the project entry point is complete.

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

## Next Steps

The current priority is converting the research code into a clean, reproducible GitHub project.

Planned improvements include:

* Migrating and documenting the core pipeline modules
* Adding a reproducible project entry point
* Adding dependency and environment documentation
* Creating a sanitized sample dataset for demonstration
* Adding example model outputs and visualizations
* Expanding the historical dataset for stronger out-of-sample validation

## Development Assistance

This project was developed with assistance from AI tools, including ChatGPT and Gemini, for code review, debugging, documentation, and implementation guidance. All research design decisions, validation logic, model evaluation, and final code review were performed and verified by the project author.
