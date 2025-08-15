import pandas as pd
import numpy as np

# Assume 'final_df' is the cleaned and merged DataFrame from the previous step.
# If you are running this in a new script, you'll need to regenerate final_df first.
# For demonstration, I'll re-include the call to the previous function.

def clean_and_merge_fpl_data(fpl_23_24_path, fpl_24_25_path, epl_24_25_path):
    """
    Loads, cleans, and merges FPL and EPL datasets into a single unified DataFrame.
    (This is the function from the previous step, included for completeness)
    """
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

    Args:
        df (pandas.DataFrame): The cleaned and merged player data.

    Returns:
        pandas.DataFrame: The DataFrame with new, engineered features.
    """
    print("\nStarting feature engineering...")

    # --- 1. Per 90 Minute Stats ---
    # Normalize stats to a "per 90 minutes" basis to compare players fairly.
    # We add a small number (1e-6) to avoid division by zero for players with 0 minutes.
    minutes_played = df['minutes']
    
    # Key FPL metrics per 90
    df['points_per_90'] = (df['total_points'] / (minutes_played + 1e-6)) * 90
    df['xg_per_90'] = (df['expected_goals'] / (minutes_played + 1e-6)) * 90
    df['xa_per_90'] = (df['expected_assists'] / (minutes_played + 1e-6)) * 90
    df['xgi_per_90'] = df['xg_per_90'] + df['xa_per_90'] # Expected Goal Involvement

    # --- 2. Value Metrics ---
    # Calculate how many points a player delivers per million pounds of their cost.
    df['points_per_million'] = df['total_points'] / df['cost']
    
    # --- 3. Team Strength Metrics ---
    # Calculate the overall attacking and defensive strength of each team.
    # This will be crucial for judging fixture difficulty.
    team_stats = df.groupby('team_name').agg(
        team_xg=('expected_goals', 'sum'),
        team_goals_scored=('goals_scored', 'sum'),
        team_xgc=('Goals Conceded', 'sum'), # Using 'Goals Conceded' from EPL stats
        team_goals_conceded=('goals_scored', 'sum') # Opponent goals scored against this team
    ).reset_index()

    # Rename for clarity
    team_stats.rename(columns={
        'team_xg': 'team_attack_strength_xg',
        'team_goals_scored': 'team_attack_strength_goals',
        'team_xgc': 'team_defense_weakness_xgc',
        'team_goals_conceded': 'team_defense_weakness_goals'
    }, inplace=True)
    
    # Merge team stats back into the main DataFrame
    df = pd.merge(df, team_stats, on='team_name', how='left')
    
    # --- 4. Historical Performance (per 90) ---
    # Do the same per-90 calculations for last season's data.
    minutes_23_24 = df['minutes_23_24']
    df['points_23_24_per_90'] = (df['points_23_24'] / (minutes_23_24 + 1e-6)) * 90
    df['xg_23_24_per_90'] = (df['xg_23_24'] / (minutes_23_24 + 1e-6)) * 90
    df['xa_23_24_per_90'] = (df['xa_23_24'] / (minutes_23_24 + 1e-6)) * 90
    
    # --- 5. Positional Indicators ---
    # Create binary flags for player positions. This can help the model learn
    # position-specific patterns.
    df['is_gk'] = (df['position'] == 'GKP').astype(int)
    df['is_def'] = (df['position'] == 'DEF').astype(int)
    df['is_mid'] = (df['position'] == 'MID').astype(int)
    df['is_fwd'] = (df['position'] == 'FWD').astype(int)

    # Fill any potential NaN values created during calculations with 0
    df.fillna(0, inplace=True)

    print("Feature engineering complete.")
    return df

# --- Execute the functions and display results ---
# First, get the cleaned DataFrame
final_df = clean_and_merge_fpl_data(
    'fpl_playerstats_2023-24.csv',
    'fpl_playerstats_2024-25.csv',
    'epl_player_stats_24_25.csv'
)

if final_df is not None:
    # Now, engineer the features
    featured_df = engineer_features(final_df)

    print("\n--- DataFrame with Engineered Features ---")
    # Displaying a subset of original and new columns for clarity
    display_cols = [
        'player_name', 'team_name', 'position', 'cost', 'total_points',
        'points_per_90', 'xgi_per_90', 'points_per_million',
        'team_attack_strength_xg', 'team_defense_weakness_xgc'
    ]
    print(featured_df[display_cols].head())

    print("\n--- Top 10 Players by Expected Goal Involvement per 90 (xgi_per_90) ---")
    print(featured_df[featured_df['minutes'] > 180].sort_values(by='xgi_per_90', ascending=False)[display_cols].head(10).to_string())
