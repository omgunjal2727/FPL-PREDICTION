FPL AI Team Optimizer ⚽🤖
This project uses a machine learning model and linear optimization to automatically select the best possible Fantasy Premier League (FPL) squad.

It analyzes player statistics, predicts future points using a pre-trained XGBoost model, and then constructs a 15-player squad that maximizes projected points while adhering to all of FPL's budget and team constraints.

Key Features
Data-Driven Predictions: Uses an XGBoost model to forecast player points based on key performance metrics.

Optimal Squad Selection: Employs PuLP, a linear programming library, to build the highest-scoring squad possible.

Constraint Adherence: The optimizer strictly follows FPL rules, including the £100m budget, player position quotas, and the 3-player-per-team limit.

Fixture Analysis: Includes fixture data to implicitly factor in the difficulty of upcoming matches.

How It Works
The project automates FPL team selection through a three-step pipeline:

Data Processing: It starts by cleaning and preparing player data from the provided CSV files (fpl_playerstats_*.csv and epl_player_stats_*.csv). This step ensures the data is ready for the prediction model.

Point Prediction: The cleaned data is fed into a pre-trained XGBoost model (fpl_xgb_model.json). The model predicts the expected FPL points for every player in the league for the upcoming Gameweek.

Team Optimization: With the player point predictions in hand, a linear programming model selects the optimal 15-player squad that maximizes the total score while satisfying all budget and team composition rules.

Tech Stack
Python: The core language for scripting and logic.

Pandas: Used for all data manipulation and preparation.

XGBoost: The machine learning library used for the predictive model.

PuLP: The linear programming library used for squad optimization.

Scikit-learn: A foundational machine learning library for Python.

How to Use This Project
To run this project on your local machine, follow these steps:

Clone the repository:

Bash

git clone <your-repository-url>
cd <your-repository-directory>
Install the required libraries:

Bash

pip install pandas xgboost pulp scikit-learn
Update the data (Optional):
Replace the .csv files in the main directory with the latest player and fixture data to get the most accurate predictions.

Run the main script:

Bash

python main.py
The script will output the optimized 15-player FPL squad to your console.

Current Limitations
This project is a powerful tool but has some key limitations:

Static Data: It relies on static CSV files. The model does not fetch live FPL data, so it cannot adapt to real-time price changes, injuries, or form fluctuations.

No Long-Term Strategy: The optimizer focuses on selecting the best team for a single gameweek and does not account for long-term planning (e.g., future fixtures, double/blank gameweeks).

No Transfer or Captaincy Logic: The tool is designed for initial squad selection and does not suggest weekly transfers or recommend the optimal captain.

Future Improvements
Integrate the official FPL API to fetch live data automatically.

Develop a multi-gameweek lookahead feature for better long-term planning.

Add a module for recommending weekly transfers and captaincy choices.

Implement model re-training to keep predictions accurate as the season progresses.







