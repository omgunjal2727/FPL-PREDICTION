import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Tuple, Optional

# ---
#
# AUTOMATED SCOUTING REPORT GENERATION (OPTIMIZED)
#
# ---

def fetch_article_text(url: str) -> str:
    """
    Fetches and extracts relevant text from a single news article URL.
    
    Args:
        url (str): The URL of the news article.

    Returns:
        str: The cleaned text content of the article.
    """
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Target specific tags that usually contain main content to get cleaner text
        main_content = soup.find('article') or soup.find('div', class_='content') or soup.body
        text = main_content.get_text(separator=' ', strip=True)
        
        # Clean up excessive whitespace
        return ' '.join(text.split())
    except requests.exceptions.RequestException as e:
        print(f"Could not fetch or parse {url}. Error: {e}")
        return ""

def get_scouting_info(player_df: pd.DataFrame) -> Optional[str]:
    """
    Searches for FPL news online and prepares a prompt for AI analysis.

    Args:
        player_df (pd.DataFrame): The DataFrame containing all player data.

    Returns:
        Optional[str]: The generated AI prompt, or None if no news is found.
    """
    print("\n--- Starting Automated Scouting ---")
    
    # --- 1. Fetch News Articles ---
    # Updated list with more specific and recent article sources
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

    # Limit the text to a reasonable size for the AI prompt
    all_article_text = all_article_text[:8000]
    
    # --- 2. Prepare Player List ---
    player_names: List[str] = player_df['player_name'].unique().tolist()

    # --- 3. Construct the AI Prompt ---
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
