import os
import pandas as pd
from joblib import load

def load_data(file_path):
    return pd.read_csv(file_path)

def get_match_features(df, home_team, away_team):
    match = df[(df['Home'] == home_team) & (df['Away'] == away_team)]
    if match.empty:
        raise ValueError("Match not found in the dataset. Please check the team names.")
    return match.iloc[0]

if __name__ == "__main__":
    # Load the trained model
    model_path = 'data/voting_classifier.pkl'
    model = load(model_path)

    # Load the engineered data (this should be the same data structure used during training)
    df = load_data('data/fe_new_matches.csv')

    # Get inputs from environment variables
    home_team = os.getenv("HOME_TEAM", "").strip().lower()
    away_team = os.getenv("AWAY_TEAM", "").strip().lower()
    match_date = os.getenv("MATCH_DATE", "").strip()
    season_end_year = os.getenv("SEASON_END_YEAR", "").strip()

    # Validate input
    if not home_team or not away_team or not match_date or not season_end_year:
        print("Home team, away team, match date, or season end year not provided.")
        exit(1)

    # Extract features for the input match
    try:
        match_features = get_match_features(df, home_team, away_team)
    except ValueError as e:
        print(e)
        exit(1)

    # Prepare features for prediction
    advanced_predictors = [
        "Venue_Code", "Opp_Code", "Day_Code", "Rolling_HomeGoals", "Rolling_AwayGoals", 
        "Venue_Opp_Interaction", "Decayed_Rolling_HomeGoals", "Decayed_Rolling_AwayGoals", 
        "Home_Advantage", "Home_Streak_Wins", "Away_Streak_Losses"
    ]

    features = match_features[advanced_predictors].to_frame().T
    prediction = model.predict(features)

    result_map = {0: "Home Win", 1: "Away Win", 2: "Draw"}
    result = result_map.get(prediction[0], "Unknown Result")
    print(f"Predicted Outcome: {result}")
    print(f"Match Date: {match_date}")
    print(f"Season End Year: {season_end_year}")
