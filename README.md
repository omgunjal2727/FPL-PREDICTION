<div align="center">

⚽ FPL AI-Powered Team Optimizer 🤖
A sophisticated analytics tool that leverages machine learning and AI-driven insights to build your optimal Fantasy Premier League squad.

</div>

<p align="center">
<img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python Version">
<img src="https://img.shields.io/badge/license-MIT-green" alt="License">
<img src="https://img.shields.io/badge/status-active-brightgreen" alt="Status">
</p>

This project moves beyond simple statistics by integrating a predictive ML model, a mathematical optimizer, and qualitative AI scouting reports to provide a decisive edge in FPL team selection. It automates the complex decision-making process, allowing you to build a high-potential team with confidence.

✨ Key Features
📊 Live Data Integration: Fetches real-time player data directly from the official FPL API, ensuring your team is based on the most current information.

🔮 Predictive Analytics: Utilizes a pre-trained XGBoost model to accurately forecast expected points (xP) for every player in the league.

🤖 AI Scouting Reports: Integrates with Google's Gemini API to generate qualitative insights on player form, injury risks, and tactical roles.

⚙️ Optimal Squad Generation: Employs the PuLP optimization library to construct the mathematically best 15-player squad within the £100m budget and all game rules.

📈 Fixture Difficulty Rating (FDR): Automatically calculates team-specific attacking and defensive strength to intelligently assess match difficulty.

🔧 Manual Overrides: Provides dictionaries in main.py for you to apply your own "scout bias" to boost or penalize players based on your intuition.

🚀 System Workflow
The project follows a clear, automated pipeline from data ingestion to final team selection:

<div align="center">

Live FPL API Data → Data Processing & FDR Calculation → XGBoost Point Prediction → Gemini AI Scouting → PuLP Squad Optimization → Final Team Output

</div>

🛠️ Technology Stack
Category

Technology

Core Language

Python 3

Data Science

Pandas, NumPy, Scikit-learn

Machine Learning

XGBoost

Optimization

PuLP

API & Web

Requests, Google Generative AI, python-dotenv

SETUP & USAGE
1. Prerequisites
Python 3.8+ and Git installed.

A Google AI Studio API Key for the AI scouting feature.

2. Installation Guide
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `.\venv\Scripts\activate`

# Install the required dependencies
pip install pandas numpy xgboost pulp requests scikit-learn python-dotenv google-generativeai

3. Configuration
Create a .env file in the project root and add your Gemini API key:

GEMINI_API_KEY="YOUR_API_KEY_HERE"

Before running, you can also modify the scout_overrides and player_team_overrides dictionaries at the top of main.py to apply your own custom logic.

4. Running the Optimizer
Execute the main script from your terminal:

python main.py

The script will output the final optimized Starting XI, Bench, and Captain to your console.

📁 Project Structure
.
├── 📂 data/
│   └── 📄 fpl_playerstats_*.csv
├── 📄 .env
├── 📄 .gitignore
├── 📄 data_processing.py
├── 📄 fixtures.py
├── 📄 fpl_xgb_model.json
├── 📄 main.py
├── 📄 scout.py
└── 📄 README.md

🎯 Limitations & Future Roadmap
Static Prediction Model: While the input data is live, the XGBoost model itself is static and does not re-train automatically as the season progresses.

Single Gameweek Focus: The optimization is performed for the immediate upcoming gameweek and lacks a long-term strategic view.

Outdated Fixture List: The hardcoded fixtures.py file may need to be updated for the current season's schedule.

Future Enhancements:

[ ] Implement a weekly model re-training pipeline.

[ ] Develop a multi-gameweek lookahead feature for strategic planning.

[ ] Add a dedicated module for suggesting weekly transfers and captaincy.
