import os
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score

def load_data(file_path):
    """Load dataset from a file."""
    return pd.read_csv(file_path)

def get_match_features(df, home_team, away_team, match_date, season_end_year):
    """Extract features for a specific match."""
    match = df[(df['Home'] == home_team) & 
               (df['Away'] == away_team) & 
               (df['Date'] == match_date) & 
               (df['Season_End_Year'] == int(season_end_year))]
    
    if match.empty:
        print(f"Match not found with the following details:")
        print(f"Home Team: {home_team}")
        print(f"Away Team: {away_team}")
        print(f"Match Date: {match_date}")
        print(f"Season End Year: {season_end_year}")
        raise ValueError("Match not found in the dataset. Please check the team names, date, and season end year.")
    
    return match.iloc[0]

def compare_team_matches(df, home_team, away_team):
    """Display past matches between the teams."""
    past_matches = df[
        ((df['Home'] == home_team) & (df['Away'] == away_team)) |
        ((df['Home'] == away_team) & (df['Away'] == home_team))
    ]

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
        actual_result = 'Home Win' if match['Target'] == 0 else ('Away Win' if match['Target'] == 1 else 'Draw')

        print(f"{match_date}: {home_team_name} {home_goals}-{away_goals} {away_team_name} (Actual: {actual_result})")

def display_available_teams(df):
    """Display available teams in the dataset to avoid input errors."""
    teams = sorted
