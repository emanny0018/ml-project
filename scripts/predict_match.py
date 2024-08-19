import os
import pandas as pd
from joblib import load

def load_data(file_path):
    """Load dataset from a file."""
    return pd.read_csv(file_path)

def get_match_features(df, home_team, away_team):
    """Extract features for a specific match."""
    print(f"Looking for match: {home_team} vs {away_team}")
    match = df[(df['Home'] == home_team) & (df['Away'] == away_team)]
    print(match)
    
    if match.empty:
        raise ValueError("Match not found in the dataset. Please check the team names.")
    
    return match.iloc[0]

def compare_team_matches(old_df, new_df, home_team, away_team):
    """Display past matches between the teams from old and new datasets."""
    past_matches = pd.concat([
        old_df[(old_df['Home'] == home_team) & (old_df['Away'] == away_team)],
        old_df[(old_df['Home'] == away_team) & (old_df['Away'] == home_team)],
        new_df[(new_df['Home'] == home_team) & (new_df['Away'] == away_team)],
        new_df[(new_df['Home'] == away_team) & (new_df['Away'] == home_team)]
    ])

    if past_matches.empty:
        print("No matches found for the specified teams.")
        return

    print("\nPast Matches:")
    for idx, match in past_matches.iterrows():
        match_date = pd.to_datetime(match['Date']).strftime('%Y-%m-%d')
        home_team_name = match['Home'].capitalize()
        away_team_name = match['Away'].capitalize()
        home_goals = match['HomeGoals']
        away_goals = match['AwayGoals']
        actual_result = 'Home Win' if match['FTR'] == 'H' else ('Away Win' if match['FTR'] == 'A' else 'Draw')

        print(f"{match_date}: {home_team_name} {home_goals}-{away_goals} {away_team_name} (Actual: {actual_result})")

if __name__ == "__main__":
    # Load the trained model
    model_path = 'data/voting_classifier.pkl'
    model = load(model_path)

    # Load the engineered data
    old_df = load_data('data/fe_old_matches.csv')
    new_df = load_data('data/fe_new_matches.csv')

    # Print available teams from fe_old_matches.csv
    print("Available teams from old matches for input:")
    available_old_teams = sorted(old_df['Home'].unique())
    for team in available_old_teams:
        print(f"  - {team}")

    # Print available teams from fe_new_matches.csv
    print("Available teams from new matches for input:")
    available_new_teams = sorted(new_df['Home'].unique())
    for team in available_new_teams:
        print(f"  - {team}")

    # Get inputs from environment variables
    home_team = os.getenv("HOME_TEAM", "").strip().lower()
    away_team = os.getenv("AWAY_TEAM", "").strip().lower()
    match_date = os.getenv("MATCH_DATE", "").strip()
    season_end_year = int(os.getenv("SEASON_END_YEAR", "").strip())

    if not home_team or not away_team or not match_date or not season_end_year:
        raise ValueError("Missing input. Ensure that HOME_TEAM, AWAY_TEAM, MATCH_DATE, and SEASON_END_YEAR are set.")

    # Compare past matches between the teams
    compare_team_matches(old_df, new_df, home_team, away_team)

    # Prepare the match info for prediction
    match_info = pd.DataFrame({
        'Home': [home_team],
        'Away': [away_team],
        'Date': [match_date],
        'Season_End_Year': [season_end_year]
    })

    # Extract features for the prediction
    features = get_match_features(new_df, home_team, away_team)

    # Define predictors based on your feature engineering process
    advanced_predictors = [
        "Venue_Code", "Opp_Code", "Day_Code", "Rolling_HomeGoals", "Rolling_AwayGoals", 
        "Venue_Opp_Interaction", "Decayed_Rolling_HomeGoals", "Decayed_Rolling_AwayGoals", 
        "Home_Advantage", "Home_Streak_Wins", "Away_Streak_Losses"
    ]

    # Select the relevant features from the match data
    features = features[advanced_predictors].to_frame().T

    # Predict the outcome
    prediction = model.predict(features)

    # Interpret the result
    result_map = {0: "Home Win", 1: "Away Win", 2: "Draw"}
    result = result_map.get(prediction[0], "Unknown Result")
    
    print(f"\nPredicted Outcome for the new match: {home_team.capitalize()} vs {away_team.capitalize()}")
    print(f"Predicted Result: {result}")
