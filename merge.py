import pandas as pd
import numpy as np

def clean_and_merge_fpl_data(fpl_23_24_path, fpl_24_25_path, epl_24_25_path):
    """
    Loads, cleans, and merges FPL and EPL datasets into a single unified DataFrame.

    Args:
        fpl_23_24_path (str): Filepath for the 2023-24 FPL player stats CSV.
        fpl_24_25_path (str): Filepath for the 2024-25 FPL player stats CSV.
        epl_24_25_path (str): Filepath for the 2024-25 EPL player stats CSV.

    Returns:
        pandas.DataFrame: A cleaned and merged DataFrame ready for feature engineering.
    """
    try:
        # FIX: Use 'utf-8-sig' encoding to handle potential Byte Order Mark (BOM)
        # in CSV files, which can cause issues with column names.
        df_23_24_fpl = pd.read_csv(fpl_23_24_path, encoding='utf-8-sig')
        df_24_25_fpl = pd.read_csv(fpl_24_25_path, encoding='utf-8-sig')
        df_24_25_epl = pd.read_csv(epl_24_25_path, encoding='utf-8-sig')
        print("All datasets loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Please check the file paths.")
        return None

    # --- 1. Standardize Team Names ---
    # Create a mapping for inconsistent team names across files
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

    # --- 2. Clean and Prepare 2024-25 FPL Data (Primary DataFrame) ---
    print("\nProcessing 2024-25 FPL data...")
    df_24_25_fpl['player_name'] = df_24_25_fpl['first_name'] + ' ' + df_24_25_fpl['second_name']
    df_24_25_fpl['team_name'] = df_24_25_fpl['team_name'].map(team_name_mapping)
    df_24_25_fpl.rename(columns={'player_position': 'position', 'player_cost': 'cost'}, inplace=True)
    df_current_season = df_24_25_fpl[['id', 'player_name', 'team_name', 'position', 'cost', 'minutes', 'total_points', 'goals_scored', 'assists', 'clean_sheets', 'expected_goals', 'expected_assists', 'influence', 'creativity', 'threat']]

    # --- 3. Clean and Prepare 2023-24 FPL Historical Data ---
    print("Processing 2023-24 FPL historical data...")
    df_23_24_fpl['team'] = df_23_24_fpl['team'].map(team_name_mapping)
    historical_cols = {
        'name': 'player_name', 'team': 'team_name', 'total_points': 'points_23_24',
        'minutes': 'minutes_23_24', 'expected_goals': 'xg_23_24', 'expected_assists': 'xa_23_24',
        'bps': 'bps_23_24', 'bonus': 'bonus_23_24', 'influence': 'influence_23_24',
        'creativity': 'creativity_23_24', 'threat': 'threat_23_24'
    }
    df_historical = df_23_24_fpl[list(historical_cols.keys())].rename(columns=historical_cols)

    # --- 4. Clean and Prepare 2024-25 EPL General Stats ---
    print("Processing 2024-25 EPL general stats...")
    df_24_25_epl = df_24_25_epl.rename(columns={'Player Name': 'player_name', 'Club': 'team_name'})
    df_24_25_epl['team_name'] = df_24_25_epl['team_name'].map(team_name_mapping)
    for col in ['Conversion %', 'Passes%', 'Crosses %', 'fThird Passes %', 'gDuels %', 'aDuels %', 'Saves %']:
        if col in df_24_25_epl.columns:
            df_24_25_epl[col] = df_24_25_epl[col].astype(str).str.replace('%', '').astype(float)

    # --- 5. Merge the DataFrames ---
    print("\nMerging DataFrames...")
    df_merged = pd.merge(df_current_season, df_historical, on=['player_name', 'team_name'], how='left')
    print(f"Shape after merging historical data: {df_merged.shape}")

    epl_cols_to_drop = ['Minutes', 'Goals', 'Assists', 'Clean Sheets', 'Yellow Cards', 'Red Cards', 'Saves', 'Penalties Saved']
    df_epl_subset = df_24_25_epl.drop(columns=[col for col in epl_cols_to_drop if col in df_24_25_epl.columns])
    df_final = pd.merge(df_merged, df_epl_subset, on=['player_name', 'team_name'], how='left')
    print(f"Shape after merging EPL stats: {df_final.shape}")

    # --- 6. Final Cleanup ---
    historical_nan_cols = [
        'points_23_24', 'minutes_23_24', 'xg_23_24', 'xa_23_24', 'bps_23_24',
        'bonus_23_24', 'influence_23_24', 'creativity_23_24', 'threat_23_24'
    ]
    for col in historical_nan_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0)

    print("\nData cleaning and merging complete.")
    return df_final

# --- Execute the function and display results ---
final_df = clean_and_merge_fpl_data(
    'fpl_playerstats_2023-24.csv',
    'fpl_playerstats_2024-25.csv',
    'epl_player_stats_24_25.csv'
)

if final_df is not None:
    print("\n--- Unified DataFrame Info ---")
    final_df.info()
    print("\n--- Unified DataFrame Head ---")
    print(final_df.head())
    print("\n--- Example row for a player with historical data ---")
    print(final_df[final_df['player_name'] == 'Bukayo Saka'].to_string())
