import requests
import pandas as pd

def scrape_fpl_data():
    """
    Scrapes the official FPL API for the latest player and team data
    and saves it to 'fpl_playerstats_2024-25.csv'.
    """
    print("Fetching latest data from the official FPL API...")
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Will raise an error for bad responses
        data = response.json()
        print("Data fetched successfully.")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching data from the API: {e}")
        return

    # Extract player, team, and position data
    players_df = pd.DataFrame(data['elements'])
    teams_df = pd.DataFrame(data['teams'])
    positions_df = pd.DataFrame(data['element_types'])

    # Map team and position names to the main player dataframe
    team_map = teams_df.set_index('id')['name']
    position_map = positions_df.set_index('id')['singular_name_short']
    
    players_df['team_name'] = players_df['team'].map(team_map)
    players_df['player_position'] = players_df['element_type'].map(position_map)
    
    # FPL API uses 'now_cost' which is cost * 10. We divide to get the actual cost.
    players_df['player_cost'] = players_df['now_cost'] / 10.0

    # Select and rename columns to match the format your script expects
    columns_to_keep = {
        'first_name': 'first_name',
        'second_name': 'second_name',
        'team_name': 'team_name',
        'player_position': 'player_position',
        'player_cost': 'player_cost',
        'minutes': 'minutes',
        'goals_scored': 'goals_scored',
        'assists': 'assists',
        'clean_sheets': 'clean_sheets',
        'bonus': 'bonus',
        'bps': 'bps',
        'influence': 'influence',
        'creativity': 'creativity',
        'threat': 'threat',
        'ict_index': 'ict_index',
        'expected_goals': 'expected_goals',
        'expected_assists': 'expected_assists',
        'expected_goal_involvements': 'expected_goal_involvements',
        'expected_goals_conceded': 'expected_goals_conceded'
    }
    
    final_df = players_df[[col for col in columns_to_keep.keys() if col in players_df.columns]]
    final_df = final_df.rename(columns=columns_to_keep)

    # Save the fresh data to the CSV file, overwriting the old one
    output_filename = 'fpl_playerstats_2024-25.csv'
    final_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"\nSuccessfully created updated player list at '{output_filename}'")
    print(f"Found {len(final_df)} players for the current season.")

if __name__ == "__main__":
    # Before running, you might need to install the 'requests' library:
    # pip install requests
    scrape_fpl_data()