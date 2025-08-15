import pandas as pd
import numpy as np
import xgboost as xgb
import requests 
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus
from fixtures import upcoming_fixtures
from scout import get_scouting_info # <-- IMPORT the new scout function

# ---
#
# SCOUT BIAS AND MANUAL OVERRIDES
#
# ---

# Use this dictionary to apply a bias based on news, expert opinion, or your own intuition.
# A score of 1.1 is a 10% boost, 0.9 is a 10% penalty. 1.0 is neutral.
scout_overrides = {
    # Player Name: (score, "Reason for bias")
    'Cole Palmer': (1.1, "Golden rule: Back the talismanic players on penalties."),
    'Erling Haaland': (1.05, "High ownership and captaincy favorite."),
    'Alexander Isak': (1.1, "On penalties and the focal point of a strong attack."),
    # Add players here who might be rotation risks or returning from injury with a score < 1.0
    # 'Player with rotation risk': (0.85, "Mentioned as a rotation risk in news articles.")
}

# If you know a player has transferred but the API hasn't updated, correct it here.
player_team_overrides = {
    'Luis Díaz': 'Bayern Munich', 
}


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
        response.raise_for_status()
        data = response.json()
        print("Live data fetched successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching live FPL data: {e}")
        return None

    players_df = pd.DataFrame(data['elements'])
    teams_df = pd.DataFrame(data['teams'])
    positions_df = pd.DataFrame(data['element_types'])

    team_map = teams_df.set_index('id')['name'].to_dict()
    position_map = positions_df.set_index('id')['singular_name_short'].to_dict()

    players_df['team_name'] = players_df['team'].map(team_map)
    players_df['position'] = players_df['element_type'].map(position_map)
    players_df['player_name'] = players_df['first_name'] + ' ' + players_df['second_name']
    players_df['cost'] = players_df['now_cost'] / 10.0

    for player_substring, new_team in player_team_overrides.items():
        mask = players_df['player_name'].str.contains(player_substring, case=False, na=False)
        if mask.any():
            full_name = players_df.loc[mask, 'player_name'].iloc[0]
            players_df.loc[mask, 'team_name'] = new_team
            print(f"Manual override applied: '{full_name}' moved to {new_team}")

    live_data_df = players_df.rename(columns={
        'total_points': 'total_points', 'goals_scored': 'goals_scored', 'assists': 'assists',
        'clean_sheets': 'clean_sheets', 'expected_goals': 'expected_goals', 'expected_assists': 'expected_assists',
        'influence': 'influence', 'creativity': 'creativity', 'threat': 'threat',
    })

    required_cols = ['id', 'player_name', 'team_name', 'position', 'cost', 'minutes', 'total_points', 'goals_scored', 'assists', 'clean_sheets', 'expected_goals', 'expected_assists', 'influence', 'creativity', 'threat']
    for col in required_cols:
        if col not in live_data_df.columns:
            live_data_df[col] = 0

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
    
    def map_team_names(df, column):
        df[column] = df[column].map(team_name_mapping).fillna(df[column])
        return df

    live_fpl_df = map_team_names(live_fpl_df, 'team_name')
    df_23_24_fpl = map_team_names(df_23_24_fpl, 'team')
    df_24_25_epl = map_team_names(df_24_25_epl, 'Club')

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

    numeric_cols = ['expected_goals', 'Goals Conceded', 'influence', 'creativity', 'threat']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float32)

    team_stats = df.groupby('team_name').agg(
        team_attack_strength=('expected_goals', 'sum'),
        team_defense_strength=('Goals Conceded', 'sum')
    ).reset_index()

    min_att, max_att = team_stats['team_attack_strength'].min(), team_stats['team_attack_strength'].max()
    min_def, max_def = team_stats['team_defense_strength'].min(), team_stats['team_defense_strength'].max()
    team_stats['attack_fdr'] = 1 + 4 * (team_stats['team_attack_strength'] - min_att) / (max_att - min_att)
    team_stats['defense_fdr'] = 1 + 4 * (team_stats['team_defense_strength'] - min_def) / (max_def - min_def)
    df = pd.merge(df, team_stats[['team_name', 'attack_fdr', 'defense_fdr']], on='team_name', how='left')

    opponent_map = df['team_name'].map(fixture_list)
    df['opponent'] = opponent_map
    df = pd.merge(df, team_stats.rename(columns={'team_name': 'opponent', 'attack_fdr': 'opponent_attack_fdr', 'defense_fdr': 'opponent_defense_fdr'}), on='opponent', how='left')
    
    is_attacker = df['position'].isin(['MID', 'FWD'])
    # FIX: Create the Series with np.where, then use the pandas .fillna() method
    fdr_series = pd.Series(np.where(is_attacker, df['opponent_defense_fdr'], df['opponent_attack_fdr']))
    df['fdr'] = fdr_series.fillna(3)


    minutes_played = df['minutes']
    df['points_per_90'] = (df['total_points'] / (minutes_played + 1e-6)) * 90
    df['xgi_per_90'] = ((pd.to_numeric(df['expected_goals'], errors='coerce').fillna(0) + pd.to_numeric(df['expected_assists'], errors='coerce').fillna(0)) / (minutes_played + 1e-6)) * 90
    
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
def train_and_predict(df, scout_bias):
    """
    Trains model and predicts expected points, adjusted for context and scout bias.
    """
    print("\nStarting model training and prediction...")
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
    
    def get_scout_score(player_name):
        for name, (score, reason) in scout_bias.items():
            if name in player_name:
                print(f"Applying scout bias for {player_name}: {score} ({reason})")
                return score
        return 1.0
    
    df['scout_score'] = df['player_name'].apply(get_scout_score)
    df['final_xP'] = df['context_adjusted_xP'] * df['scout_score']

    print("Prediction complete.")
    return df

# ---
#
# TEAM OPTIMIZATION
#
# ---
def optimize_team(df, budget=100.0):
    """
    Selects the optimal FPL squad and starting XI based on the final xP.
    """
    print("\nStarting team optimization...")
    players = df.to_dict('index')
    prob = LpProblem("FPL_Team_Optimization", LpMaximize)
    
    in_squad = LpVariable.dicts("in_squad", players.keys(), cat='Binary')
    is_starter = LpVariable.dicts("is_starter", players.keys(), cat='Binary')
    is_captain = LpVariable.dicts("is_captain", players.keys(), cat='Binary')

    prob += lpSum([players[i]['final_xP'] * (is_starter[i] + is_captain[i]) for i in players]), "Total_Final_xP"

    prob += lpSum([players[i]['cost'] * in_squad[i] for i in players]) <= budget, "Budget"
    prob += lpSum([in_squad[i] for i in players]) == 15, "Squad_Size"
    
    position_cols = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
    for pos, count in position_cols.items():
        prob += lpSum([df.loc[i, pos] * in_squad[i] for i in players]) == count
    
    for team in df['team_name'].unique():
        prob += lpSum([in_squad[i] for i in players if players[i]['team_name'] == team]) <= 3
    
    prob += lpSum([is_starter[i] for i in players]) == 11
    prob += lpSum([is_captain[i] for i in players]) == 1
    
    for i in players:
        prob += is_starter[i] <= in_squad[i]
        prob += is_captain[i] <= is_starter[i]
    
    prob += lpSum([df.loc[i, 'GKP'] * is_starter[i] for i in players]) == 1
    prob += lpSum([df.loc[i, 'DEF'] * is_starter[i] for i in players]) >= 3
    prob += lpSum([df.loc[i, 'MID'] * is_starter[i] for i in players]) >= 2
    prob += lpSum([df.loc[i, 'FWD'] * is_starter[i] for i in players]) >= 1

    prob.solve()
    print("Optimization complete. Status:", LpStatus[prob.status])
    
    if LpStatus[prob.status] == 'Optimal':
        print("\n--- AI-Selected Optimal FPL Squad ---")
        total_cost = 0; starting_xi, bench, captain = [], [], ''
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

        pos_map = {col: col for col in ['GKP', 'DEF', 'MID', 'FWD']}
        xi_df['position'] = xi_df[pos_map.keys()].idxmax(axis=1)
        bench_df['position'] = bench_df[pos_map.keys()].idxmax(axis=1)
        
        display_cols = ['player_name', 'team_name', 'position', 'cost', 'final_xP']
        print("\n--- Starting XI ---")
        print(xi_df.sort_values(by='position')[display_cols].to_string(index=False))
        print(f"\nCaptain: {captain}")
        print("\n--- Bench ---")
        print(bench_df.sort_values(by='position')[display_cols].to_string(index=False))
        
        if not xi_df.empty and captain:
            captain_player_df = xi_df[xi_df['player_name'] == captain.replace(' (C)', '')]
            if not captain_player_df.empty:
                total_predicted_points = xi_df['final_xP'].sum() + captain_player_df['final_xP'].iloc[0]
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
            scout_prompt = get_scouting_info(final_df)
            
            if scout_prompt:
                print("\n--- Generated AI Scout Prompt (for demonstration) ---")
                print(scout_prompt[:1000] + "...")
            
            featured_df = engineer_features(final_df, upcoming_fixtures)
            prediction_df = train_and_predict(featured_df, scout_overrides)
            optimize_team(prediction_df)
