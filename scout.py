import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Optional

# ---
#
# AUTOMATED SCOUTING REPORT GENERATION
#
# ---

def fetch_article_text(url: str) -> str:
    """
    Fetches and extracts relevant text from a single news article URL.
    """
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        main_content = soup.find('article') or soup.find('div', class_='content') or soup.body
        text = main_content.get_text(separator=' ', strip=True)
        return ' '.join(text.split())
    except requests.exceptions.RequestException as e:
        print(f"Could not fetch or parse {url}. Error: {e}")
        return ""

def get_scouting_info(player_df: pd.DataFrame) -> Optional[str]:
    """
    Searches for FPL news online and prepares a prompt for AI analysis.
    """
    print("\n--- Starting Automated Scouting ---")
    urls = [
        "https://www.fantasyfootballscout.co.uk/",
        "https://www.premierleague.com/en/news/4373986/scout-selection-the-best-fpl-squad-for-the-opening-gameweeks",
        "https://www.fantasyfootballfix.com/blog-index/fpl-2025-26-top-tips/",
        "https://www.premierleague.com/en/news/4373849/who-are-the-best-budget-players-for-202526-fantasy"
    ]
    
    print("Fetching news from scout websites...")
    all_article_text = " ".join([fetch_article_text(url) for url in urls])

    if not all_article_text:
        print("Could not retrieve any news. Skipping automated scouting.")
        return None

    all_article_text = all_article_text[:8000]
    player_names: List[str] = player_df['player_name'].unique().tolist()

    prompt = f"""
    Analyze the following Fantasy Premier League news text. Your task is to act as an expert FPL scout.
    Based ONLY on the text provided, identify mentions of specific players from the provided player list.
    For each player you identify, determine if the sentiment is POSITIVE (e.g., 'in form', 'nailed on', 'on penalties'), 
    NEGATIVE (e.g., 'injured', 'rotation risk', 'doubtful'), or NEUTRAL.
    Return your findings as a JSON object. The key should be the player's full name, and the value should be another
    object containing two keys: "sentiment" (either "positive", "negative", or "neutral") and "reason" (a brief quote or summary from the text).
    Do not include any players if they are not mentioned in the text.
    Here is the list of players to look for: {player_names}
    Here is the news text:
    ---
    {all_article_text}
    ---
    """
    
    print("Scouting prompt generated.")
    return prompt

def apply_scout_bias(df, scout_report):
    """
    Applies a bias to player's risk_adjusted_xP based on scouting report sentiment.
    """
    print("\nApplying scout bias...")
    if not scout_report:
        print("No scout report provided. Skipping bias application.")
        return df

    sentiment_map = {'positive': 1.1, 'negative': 0.8, 'neutral': 1.0}

    for player_name, info in scout_report.items():
        if player_name in df['player_name'].values:
            sentiment = info.get('sentiment', 'neutral').lower()
            multiplier = sentiment_map.get(sentiment, 1.0)
            reason = info.get('reason', 'No reason specified')
            df.loc[df['player_name'] == player_name, 'risk_adjusted_xP'] *= multiplier
            print(f" - Adjusted {player_name}: Multiplier={multiplier:.2f}. Reason: {reason}")
    
    print("Scout bias application complete.")
    return df
