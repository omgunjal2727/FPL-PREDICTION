import pandas as pd
import numpy as np
import pulp
import json
import os
import xgboost as xgb
from data_processing import load_and_process_data, engineer_features
from fixtures import all_gameweek_fixtures

# --- Configuration ---
FPL_STATS_PATH = 'fpl_playerstats_2024-25.csv' # This should be your 25-26 data file
MODEL_PATH = 'fpl_xgb_model.json'
BUDGET = 100.0
GAMEWEEK_TO_OPTIMIZE = 1 # Set which Gameweek to run (e.g., 1 for the first week)

# --- Manual Adjustments & Overrides ---
POSITION_OVERRIDES = {
    'Matheus Santos Carneiro Da Cunha': 'MID',
    'Rodrigo Muniz Carvalho': 'MID'
}
TEAM_EXCLUSIONS = []

# --- FIX: Add Joško Gvardiol to the exclusion list ---
PLAYER_EXCLUSIONS = [
    'Alexander Isak',
    'Joško Gvardiol',
    'Nicolas Jackson'
]

MANUAL_OVERRIDES = {
    'Cole Palmer': {'percentage': 10, 'reason': 'On penalties and team talisman.'},
    'Erling Haaland': {'percentage': 5, 'reason': 'High ownership and captaincy favorite.'}
}


def predict_points(player_data, model_path):
    print("Step 3: Predicting points with loaded model...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    try:
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        features = model.get_booster().feature_names

        for feature in features:
            if feature not in player_data.columns:
                player_data[feature] = 0
        
        prediction_input = player_data[features]
        player_data['xP'] = model.predict(prediction_input)
        
    except Exception as e:
        print(f"\nAn unexpected prediction error occurred: {e}")
        return None

    print("Prediction complete.")
    return player_data

def optimize_team(player_data, budget):
    print("Step 5: Optimizing team selection...")
    prob = pulp.LpProblem("FPL_Team_Optimization", pulp.LpMaximize)
    player_vars = {i: pulp.LpVariable(f"player_{i}", cat='Binary') for i in player_data.index}
    prob += pulp.lpSum(player_data.loc[i, 'final_xP'] * player_vars[i] for i in player_data.index)
    prob += pulp.lpSum(player_data.loc[i, 'cost'] * player_vars[i] for i in player_data.index) <= budget
    prob += pulp.lpSum(player_vars[i] for i in player_data.index) == 15
    prob += pulp.lpSum(player_vars[i] for i in player_data.index if player_data.loc[i, 'position'] == 'GKP') == 2
    prob += pulp.lpSum(player_vars[i] for i in player_data.index if player_data.loc[i, 'position'] == 'DEF') == 5
    prob += pulp.lpSum(player_vars[i] for i in player_data.index if player_data.loc[i, 'position'] == 'MID') == 5
    prob += pulp.lpSum(player_vars[i] for i in player_data.index if player_data.loc[i, 'position'] == 'FWD') == 3
    for team in player_data['team_name'].unique():
        prob += pulp.lpSum(player_vars[i] for i in player_data.index if player_data.loc[i, 'team_name'] == team) <= 3
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    print(f"Optimization complete. Status: {pulp.LpStatus[prob.status]}")
    selected_indices = [i for i in player_data.index if player_vars[i].varValue == 1]
    return player_data.loc[selected_indices]

def display_team(squad):
    squad['position_order'] = squad['position'].map({'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4})
    squad = squad.sort_values(by=['position_order', 'final_xP'], ascending=[True, False])
    
    starting_xi = pd.concat([
        squad[squad['position'] == 'GKP'].head(1),
        squad[squad['position'] == 'DEF'].head(4),
        squad[squad['position'] == 'MID'].head(4),
        squad[squad['position'] == 'FWD'].head(2)
    ])
    bench = squad.drop(starting_xi.index).sort_values(by='position_order')
    
    captain = starting_xi.loc[starting_xi['final_xP'].idxmax()]
    starting_xi_points = starting_xi['final_xP'].sum() + captain['final_xP']

    print("\n--- ✨ AI-Selected Optimal FPL Squad ✨ ---\n")
    print("--- Starting XI ---")
    print(starting_xi[['first_name', 'second_name', 'team_name', 'position', 'cost', 'final_xP']].to_string(index=False))
    print(f"\nCaptain: {captain['first_name']} {captain['second_name']} (C)\n")
    
    print("--- Bench ---")
    print(bench[['first_name', 'second_name', 'team_name', 'position', 'cost', 'final_xP']].to_string(index=False))
    
    total_cost = squad['cost'].sum()
    print(f"\nTotal Squad Cost: £{total_cost:.1f}m")
    print(f"Predicted Points for Starting XI (with Captain): {starting_xi_points:.2f}")

def main():
    try:
        fixture_list = all_gameweek_fixtures[GAMEWEEK_TO_OPTIMIZE - 1]
        print(f"Running optimization for Gameweek {GAMEWEEK_TO_OPTIMIZE}...")
    except IndexError:
        print(f"Error: Gameweek {GAMEWEEK_TO_OPTIMIZE} not found in fixtures.py.")
        return

    player_data = load_and_process_data(FPL_STATS_PATH)
    if player_data is None: return

    print("\nApplying position overrides...")
    player_data['player_name'] = player_data['first_name'] + ' ' + player_data['second_name']
    for name, pos in POSITION_OVERRIDES.items():
        if name in player_data['player_name'].values:
            player_data.loc[player_data['player_name'] == name, 'position'] = pos
            print(f" - Set {name} to position: {pos}")
            
    player_data = player_data[~player_data['team_name'].isin(TEAM_EXCLUSIONS)]
    player_data = player_data[~player_data['player_name'].isin(PLAYER_EXCLUSIONS)]
    
    player_data = engineer_features(player_data, fixture_list)
    if player_data is None: return

    player_data = predict_points(player_data, MODEL_PATH)
    if player_data is None: return
        
    player_data['final_xP'] = player_data['xP']
    
    print("\nApplying manual overrides...")
    for name, details in MANUAL_OVERRIDES.items():
        if name in player_data['player_name'].values:
            adjustment = 1 + (details['percentage'] / 100)
            player_data.loc[player_data['player_name'] == name, 'final_xP'] *= adjustment
            print(f" - Manual Override: Adjusted {name} by {details['percentage']}%. Reason: {details['reason']}")
    
    optimal_squad = optimize_team(player_data, BUDGET)
    display_team(optimal_squad)

if __name__ == "__main__":
    main()