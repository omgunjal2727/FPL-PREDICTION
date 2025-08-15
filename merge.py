import pandas as pd
import numpy as np
import xgboost as xgb
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus

# --- Functions from previous steps (collapsed for brevity) ---
def clean_and_merge_fpl_data(fpl_23_24_path, fpl_24_25_path, epl_24_25_path):
    # (This is the function from the previous step)
    try:
        df_23_24_fpl = pd.read_csv(fpl_23_24_path, encoding='utf-8-sig')
        df_24_25_fpl = pd.read_csv(fpl_24_25_path, encoding='utf-8-sig')
        df_24_25_epl = pd.read_csv(epl_24_25_path, encoding='utf-8-sig')
    except FileNotFoundError:
        return None
    team_name_mapping = {
        'Arsenal': 'Arsenal', 'Aston Villa': 'Aston Villa', 'Bournemouth': 'Bournemouth',
        'Brentford': 'Brentford', 'Brighton': 'Brighton & Hove Albion', 'Brighton & Hove Albion': 'Brighton & Hove Albion',
        'Burnley': 'Burnley', 'Chelsea': 'Chelsea', 'Crystal Palace': 'Crystal Palace',
        'Everton': 'Everton', 'Fulham': 'Fulham', 'Ipswich': 'Ipswich Town',
        'Ipswich Town': 'Ipswich Town', 'Leicester': 'Leicester City', 'Leicester City': 'Leicester City',
        'Liverpool': 'Liverpool', 'Luton': 'Luton Town', 'Luton Town': 'Luton Town',
        'Man City': 'Manchester City', 'Manchester City': 'Manchester City', 'Man Utd': 'Manchester United',
        'Manchester United': 'Manchester United', 'Newcastle': 'Newcastle United', 'Newcastle United': 'Newcastle United',
        'Nott\'m Forest': 'Nottingham Forest', 'Nottingham Forest': 'Nottingham Forest',
        'Sheffield Utd': 'Sheffield United', 'Sheffield United': 'Sheffield United', 'Southampton': 'Southampton',
        'Spurs': 'Tottenham Hotspur', 'Tottenham Hotspur': 'Tottenham Hotspur', 'West Ham': 'West Ham United',
        'West Ham United': 'West Ham United', 'Wolves': 'Wolverhampton Wanderers', 'Wolverhampton Wanderers': 'Wolverhampton Wanderers'
    }
    df_24_25_fpl['player_name'] = df_24_25_fpl['first_name'] + ' ' + df_24_25_fpl['second_name']
    df_24_25_fpl['team_name'] = df_24_25_fpl['team_name'].map(team_name_mapping)
    df_24_25_fpl.rename(columns={'player_position': 'position', 'player_cost': 'cost'}, inplace=True)
    df_current_season = df_24_25_fpl[['id', 'player_name', 'team_name', 'position', 'cost', 'minutes', 'total_points', 'goals_scored', 'assists', 'clean_sheets', 'expected_goals', 'expected_assists', 'influence', 'creativity', 'threat']]
    df_23_24_fpl['team'] = df_23_24_fpl['team'].map(team_name_mapping)
    historical_cols = {
        'name': 'player_name', 'team': 'team_name', 'total_points': 'points_23_24',
        'minutes': 'minutes_23_24', 'expected_goals': 'xg_23_24', 'expected_assists': 'xa_23_24',
        'bps': 'bps_23_24', 'bonus': 'bonus_23_24', 'influence': 'influence_23_24',
        'creativity': 'creativity_23_24', 'threat': 'threat_23_24'
    }
    df_historical = df_23_24_fpl[list(historical_cols.keys())].rename(columns=historical_cols)
    df_24_25_epl = df_24_25_epl.rename(columns={'Player Name': 'player_name', 'Club': 'team_name'})
    df_24_25_epl['team_name'] = df_24_25_epl['team_name'].map(team_name_mapping)
    for col in ['Conversion %', 'Passes%', 'Crosses %', 'fThird Passes %', 'gDuels %', 'aDuels %', 'Saves %']:
        if col in df_24_25_epl.columns:
            df_24_25_epl[col] = df_24_25_epl[col].astype(str).str.replace('%', '').astype(float)
    df_merged = pd.merge(df_current_season, df_historical, on=['player_name', 'team_name'], how='left')
    epl_cols_to_drop = ['Minutes', 'Goals', 'Assists', 'Clean Sheets', 'Yellow Cards', 'Red Cards', 'Saves', 'Penalties Saved']
    df_epl_subset = df_24_25_epl.drop(columns=[col for col in epl_cols_to_drop if col in df_24_25_epl.columns])
    df_final = pd.merge(df_merged, df_epl_subset, on=['player_name', 'team_name'], how='left')
    historical_nan_cols = [
        'points_23_24', 'minutes_23_24', 'xg_23_24', 'xa_23_24', 'bps_23_24',
        'bonus_23_24', 'influence_23_24', 'creativity_23_24', 'threat_23_24'
    ]
    for col in historical_nan_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0)
    return df_final

def engineer_features(df):
    """
    Engineers new features for the FPL player dataset.
    """
    print("\nStarting feature engineering...")

    # Fill NaN values in 'Appearances' before using it
    df['Appearances'] = df['Appearances'].fillna(0)

    # --- 1. Per 90 Minute Stats ---
    minutes_played = df['minutes']
    df['points_per_90'] = (df['total_points'] / (minutes_played + 1e-6)) * 90
    df['xg_per_90'] = (df['expected_goals'] / (minutes_played + 1e-6)) * 90
    df['xa_per_90'] = (df['expected_assists'] / (minutes_played + 1e-6)) * 90
    df['xgi_per_90'] = df['xg_per_90'] + df['xa_per_90']

    # --- 2. Value Metrics ---
    df['points_per_million'] = df['total_points'] / df['cost']
    
    # --- 3. Start Probability ---
    # A simple proxy for how likely a player is to start.
    max_games_played = df['Appearances'].max()
    if max_games_played > 0:
        df['start_probability'] = df['Appearances'] / max_games_played
    else:
        df['start_probability'] = 0

    # --- 4. Team Strength Metrics ---
    team_stats = df.groupby('team_name').agg(
        team_xg=('expected_goals', 'sum'),
        team_goals_scored=('goals_scored', 'sum'),
        team_xgc=('Goals Conceded', 'sum'),
        team_goals_conceded=('goals_scored', 'sum')
    ).reset_index()
    team_stats.rename(columns={
        'team_xg': 'team_attack_strength_xg', 'team_goals_scored': 'team_attack_strength_goals',
        'team_xgc': 'team_defense_weakness_xgc', 'team_goals_conceded': 'team_defense_weakness_goals'
    }, inplace=True)
    df = pd.merge(df, team_stats, on='team_name', how='left')
    
    # --- 5. Historical Performance (per 90) ---
    minutes_23_24 = df['minutes_23_24']
    df['points_23_24_per_90'] = (df['points_23_24'] / (minutes_23_24 + 1e-6)) * 90
    df['xg_23_24_per_90'] = (df['xg_23_24'] / (minutes_23_24 + 1e-6)) * 90
    df['xa_23_24_per_90'] = (df['xa_23_24'] / (minutes_23_24 + 1e-6)) * 90
    
    # --- 6. Positional Indicators ---
    df['is_gk'] = (df['position'] == 'GKP').astype(int)
    df['is_def'] = (df['position'] == 'DEF').astype(int)
    df['is_mid'] = (df['position'] == 'MID').astype(int)
    df['is_fwd'] = (df['position'] == 'FWD').astype(int)

    df.fillna(0, inplace=True)
    print("Feature engineering complete.")
    return df

def train_and_predict(df):
    """
    Trains an XGBoost model and predicts expected points (xP) for the next gameweek.
    """
    print("\nStarting model training and prediction...")

    features = [
        'cost', 'minutes', 'expected_goals', 'expected_assists',
        'influence', 'creativity', 'threat', 'points_23_24',
        'minutes_23_24', 'xg_23_24', 'xa_23_24', 'bps_23_24',
        'points_per_90', 'xg_per_90', 'xa_per_90', 'xgi_per_90',
        'points_per_million', 'team_attack_strength_xg',
        'team_defense_weakness_xgc', 'points_23_24_per_90',
        'xg_23_24_per_90', 'xa_23_24_per_90', 'is_gk', 'is_def',
        'is_mid', 'is_fwd', 'start_probability' # Added new feature
    ]
    target = 'points_per_90'
    X = df[features]
    y = df[target]
    
    xgbr = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    xgbr.fit(X, y)
    print("Model training complete.")

    df['xP'] = xgbr.predict(X)
    df['xP_per_match'] = (df['xP'] / 90) * 75
    # NEW: Create a risk-adjusted xP based on start probability
    df['risk_adjusted_xP'] = df['xP_per_match'] * df['start_probability']

    print("Prediction complete.")
    return df

def optimize_team(df, budget=100.0):
    """
    Selects the optimal FPL squad and starting XI based on risk-adjusted predicted points.
    """
    print("\nStarting team optimization...")
    players = df.to_dict('index')
    
    prob = LpProblem("FPL_Team_Optimization", LpMaximize)
    
    in_squad = LpVariable.dicts("in_squad", players.keys(), cat='Binary')
    is_starter = LpVariable.dicts("is_starter", players.keys(), cat='Binary')
    is_captain = LpVariable.dicts("is_captain", players.keys(), cat='Binary')

    # UPDATED: Objective function now uses 'risk_adjusted_xP'
    prob += lpSum([players[i]['risk_adjusted_xP'] * is_starter[i] for i in players]) + \
            lpSum([players[i]['risk_adjusted_xP'] * is_captain[i] for i in players]), "Total_Risk_Adjusted_xP"

    # --- Constraints (largely unchanged) ---
    prob += lpSum([players[i]['cost'] * in_squad[i] for i in players]) <= budget, "Budget"
    prob += lpSum([in_squad[i] for i in players]) == 15, "Squad_Size"
    prob += lpSum([players[i]['is_gk'] * in_squad[i] for i in players]) == 2, "Goalkeepers"
    prob += lpSum([players[i]['is_def'] * in_squad[i] for i in players]) == 5, "Defenders"
    prob += lpSum([players[i]['is_mid'] * in_squad[i] for i in players]) == 5, "Midfielders"
    prob += lpSum([players[i]['is_fwd'] * in_squad[i] for i in players]) == 3, "Forwards"
    for team in df['team_name'].unique():
        prob += lpSum([in_squad[i] for i in players if players[i]['team_name'] == team]) <= 3, f"Team_{team.replace(' ', '_')}"
    prob += lpSum([is_starter[i] for i in players]) == 11, "Starting_XI_Size"
    prob += lpSum([is_captain[i] for i in players]) == 1, "Captain_Count"
    for i in players:
        prob += is_starter[i] <= in_squad[i], f"Starter_In_Squad_{i}"
        prob += is_captain[i] <= is_starter[i], f"Captain_Is_Starter_{i}"
    prob += lpSum([players[i]['is_gk'] * is_starter[i] for i in players]) == 1, "GK_Starter"
    prob += lpSum([players[i]['is_def'] * is_starter[i] for i in players]) >= 3, "DEF_Min_Starters"
    prob += lpSum([players[i]['is_mid'] * is_starter[i] for i in players]) >= 2, "MID_Min_Starters"
    prob += lpSum([players[i]['is_fwd'] * is_starter[i] for i in players]) >= 1, "FWD_Min_Starters"

    prob.solve()
    print("Optimization complete. Status:", LpStatus[prob.status])
    
    if LpStatus[prob.status] == 'Optimal':
        print("\n--- AI-Selected Optimal FPL Squad ---")
        total_cost = 0; starting_xi = []; bench = []; captain = ''
        for i in players:
            if in_squad[i].varValue == 1:
                player = players[i]
                total_cost += player['cost']
                if is_starter[i].varValue == 1:
                    starting_xi.append(player)
                    if is_captain[i].varValue == 1:
                        captain = f"{player['player_name']} (C)"
                else:
                    bench.append(player)
        
        xi_df = pd.DataFrame(starting_xi).sort_values(by='position')
        bench_df = pd.DataFrame(bench).sort_values(by='position')
        
        # UPDATED: Display 'risk_adjusted_xP'
        display_cols = ['player_name', 'team_name', 'position', 'cost', 'risk_adjusted_xP']
        print("\n--- Starting XI ---")
        print(xi_df[display_cols].to_string(index=False))
        print(f"\nCaptain: {captain}")

        print("\n--- Bench ---")
        print(bench_df[display_cols].to_string(index=False))
        
        starting_xp = xi_df['risk_adjusted_xP'].sum()
        captain_player_name = captain.replace(' (C)', '')
        captain_xp = xi_df[xi_df['player_name'] == captain_player_name]['risk_adjusted_xP'].iloc[0]
        total_predicted_points = starting_xp + captain_xp
        
        print(f"\nTotal Squad Cost: £{total_cost:.1f}m")
        print(f"Predicted Points for Starting XI (with Captain): {total_predicted_points:.2f}")

# --- Execute the full pipeline ---
final_df = clean_and_merge_fpl_data(
    'fpl_playerstats_2023-24.csv',
    'fpl_playerstats_2024-25.csv',
    'epl_player_stats_24_25.csv'
)

if final_df is not None:
    featured_df = engineer_features(final_df)
    prediction_df = train_and_predict(featured_df)
    optimize_team(prediction_df)
