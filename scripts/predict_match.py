import pandas as pd
from joblib import load

def load_data(file_path):
    """Load dataset from a file."""
    return pd.read_csv(file_path)

def get_match_features(df, home_team, away_team):
    """Extract features for a specific match."""
    # Example logic to get features for the specific match
    # This assumes you have columns in your data that correspond to home/away teams and other features
    match = df[(df['Home'] == home_team) & (df['Away'] == away_team)]
    
    if match.empty:
        raise ValueError("Match not found in the dataset. Please check the team names.")
    
    return match.iloc[0]  # Return the first matching row as the features

if __name__ == "__main__":
    # Load the trained model
    model = load('models/model.pkl')

    # Load the engineered data (this should be the same data structure used during training)
    df = load_data('data/matches-2023-2024-engineered.csv')

    # Get user input for home and away teams
    home_team = input("Enter Home Team: ")
    away_team = input("Enter Away Team: ")

    # Extract features for the input match
    try:
        match_features = get_match_features(df, home_team, away_team)
    except ValueError as e:
        print(e)
        exit(1)

    # Prepare features for prediction
    advanced_predictors = [
        "Venue_Code", "Opp_Code", "Day_Code", "Rolling_HomeGoals", "Rolling_AwayGoals", 
        "Venue_Opp_Interaction", "Recent_Form_Home", "Recent_Form_Away", 
        "Decayed_Rolling_HomeGoals", "Decayed_Rolling_AwayGoals", "Home_Advantage",
        'Home_Streak_Wins', 'Away_Streak_Losses'
    ]

    # Select the relevant features from the match data
    features = match_features[advanced_predictors].to_frame().T

    # Predict the outcome
    prediction = model.predict(features)

    # Interpret the result
    result_map = {0: "Home Win", 1: "Away Win", 2: "Draw"}
    result = result_map.get(prediction[0], "Unknown Result")
    
    print(f"Predicted Outcome: {result}")

