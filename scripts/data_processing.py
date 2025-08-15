# data_processing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --- NEW: Define the official 20 teams for the 2024-25 Premier League season ---
PREMIER_LEAGUE_TEAMS = [
    'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton & Hove Albion',
    'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich Town',
    'Leicester City', 'Liverpool', 'Manchester City', 'Manchester United',
    'Newcastle United', 'Nottingham Forest', 'Southampton', 'Tottenham Hotspur',
    'West Ham United', 'Wolverhampton Wanderers'
]

TEAM_NAME_MAPPING = {
    'Arsenal': 'Arsenal', 'Aston Villa': 'Aston Villa', 'Bournemouth': 'Bournemouth',
    'Brentford': 'Brentford', 'Brighton': 'Brighton & Hove Albion',
    'Chelsea': 'Chelsea', 'Crystal Palace': 'Crystal Palace',
    'Everton': 'Everton', 'Fulham': 'Fulham', 'Ipswich': 'Ipswich Town',
    'Leicester': 'Leicester City', 'Liverpool': 'Liverpool',
    'Man City': 'Manchester City', 'Man Utd': 'Manchester United',
    'Newcastle': 'Newcastle United', 'Nott\'m Forest': 'Nottingham Forest',
    'Southampton': 'Southampton', 'Spurs': 'Tottenham Hotspur',
    'West Ham': 'West Ham United', 'Wolves': 'Wolverhampton Wanderers'
}

def load_and_process_data(fpl_data_path):
    """
    Loads the single, up-to-date FPL data file and prepares it for feature engineering.
    """
    print("Step 1: Loading ONLY the fresh FPL data...")
    try:
        df = pd.read_csv(fpl_data_path, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"Error: Could not find the file at {fpl_data_path}")
        print("Please ensure you have run 'scraping_script.py' first.")
        return None

    df.rename(columns={'player_position': 'position', 'player_cost': 'cost'}, inplace=True)
    df['team_name'] = df['team_name'].map(TEAM_NAME_MAPPING).fillna(df['team_name'])
    
    # Filter to only include current PL teams
    df = df[df['team_name'].isin(PREMIER_LEAGUE_TEAMS)]
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(0)

    print("Data loading complete.")
    return df

def engineer_features(df, fixture_list):
    """
    Engineers new features and, most importantly, SCALES the data for the model.
    """
    print("Step 2: Engineering features and scaling data...")
    if df is None or df.empty:
        return None

    features_to_scale = [
        'influence', 'creativity', 'threat', 'bps',
        'expected_goals', 'expected_assists', 'ict_index',
        'expected_goals_conceded'
    ]
    features_to_scale = [col for col in features_to_scale if col in df.columns]
    
    if features_to_scale:
        scaler = MinMaxScaler()
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
        print("Data scaling applied successfully.")

    team_stats = df.groupby('team_name').agg(
        team_attack_strength=('expected_goals', 'sum'),
        team_defense_strength=('expected_goals_conceded', 'sum')
    ).reset_index()

    for stat in ['team_attack_strength', 'team_defense_strength']:
        min_val, max_val = team_stats[stat].min(), team_stats[stat].max()
        fdr_col = 'attack_fdr' if 'attack' in stat else 'defense_fdr'
        if max_val > min_val:
            team_stats[fdr_col] = 1 + 4 * (team_stats[stat] - min_val) / (max_val - min_val)
        else:
            team_stats[fdr_col] = 3

    df = pd.merge(df, team_stats[['team_name', 'attack_fdr', 'defense_fdr']], on='team_name', how='left')
    df['opponent'] = df['team_name'].map(fixture_list)
    opponent_stats = team_stats.rename(columns={
        'team_name': 'opponent', 'attack_fdr': 'opponent_attack_fdr', 'defense_fdr': 'opponent_defense_fdr'
    })
    df = pd.merge(df, opponent_stats, on='opponent', how='left')
    df.fillna(3, inplace=True)

    df['fdr'] = np.where(df['position'].isin(['MID', 'FWD']), df['opponent_defense_fdr'], df['opponent_attack_fdr'])
    
    print("Feature engineering complete.")
    return df