import requests
import pandas as pd
import xgboost as xgb
import numpy as np
from itertools import combinations

# --- USER INPUTS: PLEASE UPDATE THESE VALUES ---
# To find your Team ID, log in to the FPL website and go to the "Points" tab.
# The ID will be in the URL (e.g., https://fantasy.premierleague.com/entry/YOUR_ID/event/1)
TEAM_ID = 9247011  # <--- REPLACE WITH YOUR FPL TEAM ID
MONEY_IN_BANK = 0.0 # <--- REPLACE WITH YOUR MONEY IN THE BANK (e.g., 1.5 for £1.5m)
# --- END OF USER INPUTS ---

# --- API Endpoints ---
FPL_API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
USER_TEAM_URL = f"https://fantasy.premierleague.com/api/entry/9247011/event/1/picks/"

# --- Model and Data Paths ---
MODEL_PATH = 'data/fpl_xgb_model.json'

def get_fpl_data():
    """Fetches and processes live data from the FPL API."""
    print("Fetching live FPL data...")
    response = requests.get(FPL_API_URL)
    response.raise_for_status()
    data = response.json()

    players_df = pd.DataFrame(data['elements'])
    teams_df = pd.DataFrame(data['teams'])
    
    team_name_map = teams_df.set_index('id')['name']
    position_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    players_df['team_name'] = players_df['team'].map(team_name_map)
    players_df['position'] = players_df['element_type'].map(position_map)
    players_df['cost'] = players_df['now_cost'] / 10.0
    
    print("Data fetched and processed successfully.")
    return players_df, teams_df

# --- MODIFIED FUNCTION ---
def predict_points(players_df):
    """Predicts expected points (xP) for all players using the pre-trained model."""
    print(f"Loading XGBoost model from '{MODEL_PATH}'...")
    try:
        model = xgb.Booster()
        model.load_model(MODEL_PATH)
    except xgb.core.XGBoostError as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'fpl_xgb_model.json' is in the 'data/' directory.")
        return None

    # --- FIX 1: Use the exact feature list from the model's training ---
    # The error message told us exactly what the model expects.
    features = [
        'now_cost', 'minutes', 'influence', 'creativity', 'threat', 'bps', 
        'bonus', 'GKP', 'DEF', 'MID', 'FWD'
    ]
    
    # --- FIX 2: Add the one-hot encoded position columns (GKP, DEF, etc.) ---
    # Create the position columns and set them to 0
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        players_df[pos] = 0
    
    # Set the correct position column to 1 for each player
    players_df.loc[players_df['position'] == 'GKP', 'GKP'] = 1
    players_df.loc[players_df['position'] == 'DEF', 'DEF'] = 1
    players_df.loc[players_df['position'] == 'MID', 'MID'] = 1
    players_df.loc[players_df['position'] == 'FWD', 'FWD'] = 1
    
    # Ensure all feature columns are numeric and handle missing values
    for col in ['influence', 'creativity', 'threat', 'bps', 'bonus']:
         players_df[col] = pd.to_numeric(players_df[col], errors='coerce').fillna(0)

    print("Predicting points for all players...")
    # Ensure the DataFrame columns are in the exact order the model expects
    dmatrix = xgb.DMatrix(players_df[features])
    
    predictions = model.predict(dmatrix)
    players_df['xP'] = predictions
    players_df['xP'] = players_df['xP'].clip(lower=0)
    
    print("Point prediction complete.")
    return players_df

def get_user_squad(players_df):
    """Fetches the user's current squad from the FPL API."""
    print(f"Fetching your squad for Team ID: {TEAM_ID}...")
    try:
        response = requests.get(USER_TEAM_URL)
        response.raise_for_status()
        squad_data = response.json()
    except requests.exceptions.HTTPError as e:
        print(f"\nError fetching your team. The Team ID '{TEAM_ID}' might be incorrect or private.")
        print(f"API Error: {e}")
        return None

    squad_player_ids = [player['element'] for player in squad_data['picks']]
    user_squad_df = players_df[players_df['id'].isin(squad_player_ids)].copy()
    user_squad_df['selling_price'] = user_squad_df['cost']
    return user_squad_df

def suggest_transfers(user_squad_df, all_players_df, money_in_bank):
    """Analyzes and suggests the best single transfer."""
    print("Analyzing potential transfers...")
    potential_transfers = []
    initial_team_counts = user_squad_df['team_name'].value_counts()
    buy_pool = all_players_df[~all_players_df['id'].isin(user_squad_df['id'])]

    for _, player_out in user_squad_df.iterrows():
        budget = player_out['selling_price'] + money_in_bank
        replacements = buy_pool[
            (buy_pool['position'] == player_out['position']) &
            (buy_pool['cost'] <= budget)
        ].copy()

        for _, player_in in replacements.iterrows():
            team_out = player_out['team_name']
            team_in = player_in['team_name']
            
            if team_out != team_in and initial_team_counts.get(team_in, 0) >= 3:
                continue
            
            xp_gain = player_in['xP'] - player_out['xP']
            
            if xp_gain > 0.25:
                potential_transfers.append({
                    'player_out': f"{player_out['web_name']} ({player_out['team_name']})",
                    'player_in': f"{player_in['web_name']} ({player_in['team_name']})",
                    'xp_gain': xp_gain,
                    'out_xp': player_out['xP'],
                    'in_xp': player_in['xP'],
                    'out_cost': player_out['selling_price'],
                    'in_cost': player_in['cost']
                })
    
    return sorted(potential_transfers, key=lambda x: x['xp_gain'], reverse=True)

def main():
    """Main function to run the transfer suggestion process."""
    all_players, teams = get_fpl_data()
    
    players_with_xp = predict_points(all_players.copy())
    if players_with_xp is None:
        return

    user_squad = get_user_squad(players_with_xp)
    if user_squad is None:
        return
    
    suggestions = suggest_transfers(user_squad, players_with_xp, MONEY_IN_BANK)
    
    print("\n" + "="*50)
    print("✅ Top 5 Transfer Suggestions ✅")
    print("="*50)
    
    if not suggestions:
        print("\nNo beneficial transfers found. Your team looks optimal for now!")
    else:
        for i, transfer in enumerate(suggestions[:5]):
            print(f"\n--- Suggestion #{i+1} ---")
            print(f"   SELL: {transfer['player_out']} (Cost: £{transfer['out_cost']:.1f}m, xP: {transfer['out_xp']:.2f})")
            print(f"   BUY: {transfer['player_in']} (Cost: £{transfer['in_cost']:.1f}m, xP: {transfer['in_xp']:.2f})")
            print(f"  Point Gain: +{transfer['xp_gain']:.2f}")

if __name__ == "__main__":
    main()