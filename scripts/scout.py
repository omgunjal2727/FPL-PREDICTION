# scout.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Optional, Dict
import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv() # Load environment variables from .env file

# ---
#
# FULLY FUNCTIONAL AUTOMATED SCOUTING REPORT
#
# ---

def fetch_article_text(url: str) -> str:
    """Fetches and extracts text from a news article URL."""
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = ' '.join(p.get_text() for p in soup.find_all('p'))
        return ' '.join(text.split())
    except requests.exceptions.RequestException as e:
        print(f"Could not fetch or parse {url}. Error: {e}")
        return ""

def get_ai_scout_report(player_df: pd.DataFrame) -> Optional[Dict]:
    """
    Fetches FPL news, prepares a prompt, calls the Gemini AI,
    and returns a structured scouting report.
    """
    print("Step 4: Running automated scout...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found. Skipping AI scouting.")
        return None
    
    genai.configure(api_key=api_key)
    
    # === FIX: Use the latest, correct model name ===
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    urls = [
        "https://www.fantasyfootballscout.co.uk/",
        "https://www.premierleague.com/news" # A more general news source
    ]
    
    print("Fetching news from scout websites...")
    all_article_text = " ".join([fetch_article_text(url) for url in urls])

    if not all_article_text:
        print("Could not retrieve any news. Skipping AI scouting.")
        return None

    # We only care about players likely to play
    relevant_players = player_df[player_df['minutes'] > 0]['player_name'].unique().tolist()

    prompt = f"""
    Analyze the following Fantasy Premier League news text. Your task is to act as an expert FPL scout.
    Based ONLY on the text provided, identify mentions of specific players from the provided player list.
    For each player you identify, determine if the sentiment is POSITIVE (e.g., 'in form', 'nailed on', 'on penalties'), 
    NEGATIVE (e.g., 'injured', 'rotation risk', 'doubtful'), or NEUTRAL.
    Return your findings as a JSON object ONLY. The key must be the player's full name, and the value should be another
    object containing two keys: "sentiment" (either "positive", "negative", or "neutral") and "reason" (a brief quote or summary from the text).
    Do not include players if they are not mentioned. Do not include any text before or after the JSON object.

    Player List: {relevant_players[:100]}
    News Text:
    ---
    {all_article_text[:6000]}
    ---
    """
    
    response_text = "" # Initialize response_text to handle potential errors
    try:
        print("Sending prompt to AI for analysis...")
        response = model.generate_content(prompt)
        response_text = response.text # Store the text here
        # Clean up the response to extract only the JSON part
        json_response_text = response_text.strip().replace('```json', '').replace('```', '').strip()
        scout_report = json.loads(json_response_text)
        print("AI scout report received successfully.")
        return scout_report
    except Exception as e:
        print(f"Error processing AI response: {e}")
        # === FIX: Safely print the raw response text if it exists ===
        if response_text:
            print("Raw response:", response_text)
        return None

def apply_scout_bias(df: pd.DataFrame, scout_report: Dict, final_xp_col: str = 'final_xP') -> pd.DataFrame:
    """
    Applies a bias to player's xP based on the AI scouting report sentiment.
    """
    print("\nApplying AI scout bias...")
    if not scout_report:
        print("No scout report provided. Skipping bias application.")
        return df

    sentiment_map = {'positive': 1.15, 'negative': 0.80, 'neutral': 1.0}

    for player_name, info in scout_report.items():
        if player_name in df['player_name'].values:
            sentiment = info.get('sentiment', 'neutral').lower()
            multiplier = sentiment_map.get(sentiment, 1.0)
            reason = info.get('reason', 'No reason specified')
            
            df.loc[df['player_name'] == player_name, final_xp_col] *= multiplier
            print(f" - AI Scout: Adjusted {player_name} by {int((multiplier-1)*100)}%. Reason: {reason}")
    
    print("AI scout bias application complete.")
    return df