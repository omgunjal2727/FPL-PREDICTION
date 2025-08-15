# model_trainer.py

import pandas as pd
import xgboost as xgb
from data_processing import load_and_merge_data, engineer_features

# Define file paths for training data
FPL_23_24_PATH = 'fpl_playerstats_2023-24.csv'
FPL_24_25_PATH = 'fpl_playerstats_2024-25.csv' # Use current data as basis for features
EPL_24_25_PATH = 'epl_player_stats_24_25.csv'
MODEL_SAVE_PATH = 'fpl_xgb_model.json'

def train_model():
    """
    Loads historical data, engineers features, trains an XGBoost model,
    and saves it to a file.
    """
    print("--- Starting Model Training ---")
    
    # We use the 2023-24 data as our training ground.
    # We can't use live fixtures for training, so we pass an empty dict.
    df_train = pd.read_csv(FPL_23_24_PATH, encoding='utf-8-sig')
    
    # Basic feature engineering for training
    df_train['player_name'] = df_train['name']
    df_train['position'] = df_train['position']
    df_train = pd.get_dummies(df_train, columns=['position'], prefix='', prefix_sep='')

    # Define features and target
    # These should be columns that were available in the 23-24 dataset
    features_model = [
        'now_cost', 'minutes', 'influence', 'creativity', 'threat',
        'bps', 'bonus', 'GKP', 'DEF', 'MID', 'FWD'
    ]
    target = 'total_points'

    # Ensure all feature columns exist, fill with 0 if not
    for col in features_model:
        if col not in df_train.columns:
            df_train[col] = 0
            
    X_train = df_train[features_model]
    y_train = df_train[target]
    
    # Initialize and train the XGBoost Regressor
    xgbr = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    print("Training XGBoost model...")
    xgbr.fit(X_train, y_train)
    
    # Save the trained model to a file
    xgbr.save_model(MODEL_SAVE_PATH)
    print(f"Model training complete. Model saved to '{MODEL_SAVE_PATH}'")

if __name__ == "__main__":
    train_model()