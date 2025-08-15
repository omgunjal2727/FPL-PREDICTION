import pandas as pd
import numpy as np
import xgboost as xgb
import requests # <-- NEW library for fetching live data
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus
from fixtures import upcoming_fixtures

# ---
#
# LIVE DATA FETCHING
#
# ---
def get_live_fpl_data():
    """
    Fetches the latest player and team data from the official FPL API.
    """
    print("Fetching live data from FPL API...")
    try:
        url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        print("Live data fetched successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching live FPL data: {e}")
        return None

    # Create DataFrames from the JSON data
    players_df = pd.DataFrame(data['elements'])
    teams_df = pd.DataFrame(data['teams'])
    positions_df = pd.DataFrame(data['element_types'])

    # Map team and position IDs to names
    team_map = teams_df.set_index('id')['name'].to_dict()
    position_map = positions_df.set_index('id')['singular_name_short'].to_dict()

    players_df['team_name'] = players_df['team'].map(team_map)
    players_df['position'] = players_df['element_type'].map(position_map)

    # Clean up the DataFrame to match our expected format
    players_df['player_name'] = players_df['first_name'] + ' ' + players_df['second_name']
    players_df['cost'] = players_df['now_cost'] / 10.0 # API cost is 10x the actual value

    # --- NEW: Manual Overrides for outdated API data ---
    # If you know a player has transferred but the API hasn't updated, correct it here.
    player_team_overrides = {
        'Luis Díaz': 'Bayern Munich', # Example: Correcting Luis Diaz's team
        # 'Another Player': 'Their New Correct Team'
    }
    
    # FIX: Use a more robust partial string match for the override
    for player_substring, new_team in player_team_overrides.items():
        # Find rows where the player_name contains the substring
        mask = players_df['player_name'].str.contains(player_substring, case=False, na=False)
        if mask.any():
            # Get the full name for a more informative message
            full_name = players_df.loc[mask, 'player_name'].iloc[0]
            players_df.loc[mask, 'team_name'] = new_team
            print(f"Manual override applied: '{full_name}' moved to {new_team}")


    # Select and rename columns to be consistent with our script
    live_data_df = players_df.rename(columns={
        'total_points': 'total_points',
        'goals_scored': 'goals_scored',
        'assists': 'assists',
        'clean_sheets': 'clean_sheets',
        'expected_goals': 'expected_goals',
        'expected_assists': 'expected_assists',
        'influence': 'influence',
        'creativity': 'creativity',
        'threat': 'threat',
    })

    # Ensure all required columns exist, even if not in the API for some reason
    required_cols = ['id', 'player_name', 'team_name', 'position', 'cost', 'minutes', 'total_points', 'goals_scored', 'assists', 'clean_sheets', 'expected_goals', 'expected_assists', 'influence', 'creativity', 'threat']
    for col in required_cols:
        if col not in live_data_df.columns:
            live_data_df[col] = 0 # Default to 0 if a stat is missing

    return live_data_df[required_cols]


# ---
#
# DATA CLEANING AND MERGING
#
# ---
def clean_and_merge_fpl_data(live_fpl_df, fpl_23_24_path, epl_24_25_path):
    """
    Loads, cleans, and merges FPL and EPL datasets into a single unified DataFrame.
    """
    try:
        df_23_24_fpl = pd.read_csv(fpl_23_24_path, encoding='utf-8-sig')
        df_24_25_epl = pd.read_csv(epl_24_25_path, encoding='utf-8-sig')
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure all CSV files are in the correct directory.")
        return None

    team_name_mapping = {
        'Arsenal': 'Arsenal', 'Aston Villa': 'Aston Villa', 'Bournemouth': 'Bournemouth',
        'Brentford': 'Brentford', 'Brighton': 'Brighton & Hove Albion',
        'Burnley': 'Burnley', 'Chelsea': 'Chelsea', 'Crystal Palace': 'Crystal Palace',
        'Everton': 'Everton', 'Fulham': 'Fulham', 'Ipswich': 'Ipswich Town',
        'Leicester': 'Leicester City', 'Liverpool': 'Liverpool', 'Luton': 'Luton Town',
        'Man City': 'Manchester City', 'Man Utd': 'Manchester United',
        'Newcastle': 'Newcastle United', 'Nott\'m Forest': 'Nottingham Forest',
        'Sheffield Utd': 'Sheffield United', 'Southampton': 'Southampton',
        'Spurs': 'Tottenham Hotspur', 'West Ham': 'West Ham United',
        'Wolves': 'Wolverhampton Wanderers'
    }
    
    # Standardize team names across all dataframes
    live_fpl_df['team_name'] = live_fpl_df['team_name'].map(team_name_mapping).fillna(live_fpl_df['team_name'])
    df_23_24_fpl['team'] = df_23_24_fpl['team'].map(team_name_mapping).fillna(df_23_24_fpl['team'])
    df_24_25_epl['Club'] = df_24_25_epl['Club'].map(team_name_mapping).fillna(df_24_25_epl['Club'])

    historical_cols = {'name': 'player_name', 'team': 'team_name', 'total_points': 'points_23_24', 'minutes': 'minutes_23_24'}
    df_historical = df_23_24_fpl[list(historical_cols.keys())].rename(columns=historical_cols)

    df_24_25_epl = df_24_25_epl.rename(columns={'Player Name': 'player_name', 'Club': 'team_name'})

    df_merged = pd.merge(live_fpl_df, df_historical, on=['player_name', 'team_name'], how='left')
    df_final = pd.merge(df_merged, df_24_25_epl, on=['player_name', 'team_name'], how='left')
    df_final[['points_23_24', 'minutes_23_24']] = df_final[['points_23_24', 'minutes_23_24']].fillna(0)
    return df_final

# ---
#
# FEATURE ENGINEERING
#
# ---
def engineer_features(df, fixture_list):
    """
    Engineers new features for the FPL player dataset, including fixture difficulty.
    """
    print("\nStarting feature engineering...")
    df['Appearances'] = df['Appearances'].fillna(0)

    # FIX: Ensure key statistical columns are numeric before calculations
    numeric_cols = ['expected_goals', 'Goals Conceded', 'influence', 'creativity', 'threat']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Team Strength Metrics
    team_stats = df.groupby('team_name').agg(
        team_attack_strength=('expected_goals', 'sum'),
        team_defense_strength=('Goals Conceded', 'sum')
    ).reset_index()

    min_att, max_att = team_stats['team_attack_strength'].min(), team_stats['team_attack_strength'].max()
    min_def, max_def = team_stats['team_defense_strength'].min(), team_stats['team_defense_strength'].max()
    team_stats['attack_fdr'] = 1 + 4 * (team_stats['team_attack_strength'] - min_att) / (max_att - min_att)
    team_stats['defense_fdr'] = 1 + 4 * (team_stats['team_defense_strength'] - min_def) / (max_def - min_def)
    df = pd.merge(df, team_stats[['team_name', 'attack_fdr', 'defense_fdr']], on='team_name', how='left')

    # Fixture Difficulty Rating (FDR)
    def get_fdr(row):
        opponent = fixture_list.get(row['team_name'])
        if not opponent: return 3
        opponent_stats = team_stats[team_stats['team_name'] == opponent]
        if opponent_stats.empty: return 3
        return opponent_stats['defense_fdr'].iloc[0] if row['position'] in ['MID', 'FWD'] else opponent_stats['attack_fdr'].iloc[0]
    df['fdr'] = df.apply(get_fdr, axis=1)

    # Per 90 & Value Metrics
    minutes_played = df['minutes']
    df['points_per_90'] = (df['total_points'] / (minutes_played + 1e-6)) * 90
    df['xgi_per_90'] = ((pd.to_numeric(df['expected_goals'], errors='coerce').fillna(0) + pd.to_numeric(df['expected_assists'], errors='coerce').fillna(0)) / (minutes_played + 1e-6)) * 90
    
    # Improved Nailedness Score
    max_minutes = df['Appearances'].max() * 90
    df['nailedness_score'] = (df['minutes'] / max_minutes).clip(0, 1) if max_minutes > 0 else 0

    df.fillna(0, inplace=True)
    print("Feature engineering complete.")
    return df

# ---
#
# MODEL TRAINING AND PREDICTION
#
# ---
def train_and_predict(df):
    """
    Trains model and predicts expected points, adjusted for context.
    """
    print("\nStarting model training and prediction...")
    # One-hot encode position for the model
    df = pd.get_dummies(df, columns=['position'], prefix='', prefix_sep='')
    
    features_model = [
        'cost', 'minutes', 'influence', 'creativity', 'threat',
        'points_per_90', 'xgi_per_90', 'attack_fdr', 'defense_fdr', 'fdr',
        'nailedness_score', 'GKP', 'DEF', 'MID', 'FWD'
    ]
    
    for col in features_model:
        if col not in df.columns: df[col] = 0

    target = 'points_per_90'
    X = df[features_model]
    y = df[target]
    
    xgbr = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    xgbr.fit(X, y)
    print("Model training complete.")

    df['xP_per_match'] = (xgbr.predict(X) / 90) * 80
    df['context_adjusted_xP'] = df['xP_per_match'] * df['nailedness_score'] * (6 - df['fdr']) / 3
    
    print("Prediction complete.")
    return df

# ---
#
# TEAM OPTIMIZATION
#
# ---
def optimize_team(df, budget=100.0):
    """
    Selects the optimal FPL squad and starting XI.
    """
    print("\nStarting team optimization...")
    players = df.to_dict('index')
    prob = LpProblem("FPL_Team_Optimization", LpMaximize)
    
    in_squad = LpVariable.dicts("in_squad", players.keys(), cat='Binary')
    is_starter = LpVariable.dicts("is_starter", players.keys(), cat='Binary')
    is_captain = LpVariable.dicts("is_captain", players.keys(), cat='Binary')

    prob += lpSum([players[i]['context_adjusted_xP'] * (is_starter[i] + is_captain[i]) for i in players]), "Total_xP"

    # Constraints
    prob += lpSum([players[i]['cost'] * in_squad[i] for i in players]) <= budget, "Budget"
    prob += lpSum([in_squad[i] for i in players]) == 15, "Squad_Size"
    prob += lpSum([df.loc[i, 'GKP'] * in_squad[i] for i in players]) == 2, "Goalkeepers"
    prob += lpSum([df.loc[i, 'DEF'] * in_squad[i] for i in players]) == 5, "Defenders"
    prob += lpSum([df.loc[i, 'MID'] * in_squad[i] for i in players]) == 5, "Midfielders"
    prob += lpSum([df.loc[i, 'FWD'] * in_squad[i] for i in players]) == 3, "Forwards"
    for team in df['team_name'].unique():
        prob += lpSum([in_squad[i] for i in players if players[i]['team_name'] == team]) <= 3, f"Team_{team.replace(' ', '_')}"
    prob += lpSum([is_starter[i] for i in players]) == 11, "Starting_XI_Size"
    prob += lpSum([is_captain[i] for i in players]) == 1, "Captain_Count"
    for i in players:
        prob += is_starter[i] <= in_squad[i]
        prob += is_captain[i] <= is_starter[i]
    prob += lpSum([df.loc[i, 'GKP'] * is_starter[i] for i in players]) == 1, "GK_Starter"
    prob += lpSum([df.loc[i, 'DEF'] * is_starter[i] for i in players]) >= 3
    prob += lpSum([df.loc[i, 'MID'] * is_starter[i] for i in players]) >= 2
    prob += lpSum([df.loc[i, 'FWD'] * is_starter[i] for i in players]) >= 1

    prob.solve()
    print("Optimization complete. Status:", LpStatus[prob.status])
    
    if LpStatus[prob.status] == 'Optimal':
        print("\n--- AI-Selected Optimal FPL Squad ---")
        total_cost = 0
        starting_xi, bench, captain = [], [], ''
        for i in players:
            if in_squad[i].varValue == 1:
                player = df.loc[i].to_dict()
                total_cost += player['cost']
                if is_starter[i].varValue == 1:
                    starting_xi.append(player)
                    if is_captain[i].varValue == 1: captain = f"{player['player_name']} (C)"
                else: bench.append(player)
        
        xi_df = pd.DataFrame(starting_xi)
        bench_df = pd.DataFrame(bench)

        def get_pos(row):
            if row.get('GKP') == 1: return 'GKP'
            if row.get('DEF') == 1: return 'DEF'
            if row.get('MID') == 1: return 'MID'
            if row.get('FWD') == 1: return 'FWD'
        xi_df['position'] = xi_df.apply(get_pos, axis=1)
        bench_df['position'] = bench_df.apply(get_pos, axis=1)
        
        display_cols = ['player_name', 'team_name', 'position', 'cost', 'context_adjusted_xP']
        print("\n--- Starting XI ---")
        print(xi_df.sort_values(by='position')[display_cols].to_string(index=False))
        print(f"\nCaptain: {captain}")
        print("\n--- Bench ---")
        print(bench_df.sort_values(by='position')[display_cols].to_string(index=False))
        
        if not xi_df.empty and captain:
            captain_player_df = xi_df[xi_df['player_name'] == captain.replace(' (C)', '')]
            if not captain_player_df.empty:
                total_predicted_points = xi_df['context_adjusted_xP'].sum() + captain_player_df['context_adjusted_xP'].iloc[0]
                print(f"\nTotal Squad Cost: £{total_cost:.1f}m")
                print(f"Predicted Points for Starting XI (with Captain): {total_predicted_points:.2f}")

# ---
#
# EXECUTE THE FULL PIPELINE
#
# ---
if __name__ == "__main__":
    live_data = get_live_fpl_data()
    
    if live_data is not None:
        final_df = clean_and_merge_fpl_data(
            live_data,
            'fpl_playerstats_2023-24.csv',
            'epl_player_stats_24_25.csv'
        )

        if final_df is not None:
            featured_df = engineer_features(final_df, upcoming_fixtures)
            prediction_df = train_and_predict(featured_df)
            optimize_team(prediction_df)
