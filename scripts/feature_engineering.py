import pandas as pd
import numpy as np

def calculate_streaks(df):
    """Calculate win/loss streaks."""
    df['Win'] = (df['Target'] == 0).astype(int)
    df['Loss'] = (df['Target'] == 1).astype(int)
    
    df['Home_Streak_Wins'] = df.groupby('Home')['Win'].cumsum() - df.groupby('Home')['Win'].cummax().shift().fillna(0)
    df['Away_Streak_Losses'] = df.groupby('Away')['Loss'].cumsum() - df.groupby('Away')['Loss'].cummax().shift().fillna(0)
    
    return df

def add_features(df, dataset_type, old_streaks=None):
    if dataset_type == 'old':
        df["Venue_Code"] = df["Home"].astype("category").cat.codes
        df["Opp_Code"] = df["Away"].astype("category").cat.codes
        df["Day_Code"] = pd.to_datetime(df["Date"]).dt.dayofweek
        df["Rolling_HomeGoals"] = df.groupby("Home")["HomeGoals"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df["Rolling_AwayGoals"] = df.groupby("Away")["AwayGoals"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df["Venue_Opp_Interaction"] = df["Venue_Code"] * df["Opp_Code"]
        df["Decayed_Rolling_HomeGoals"] = df.groupby('Home')['HomeGoals'].transform(lambda x: x.ewm(alpha=0.9).mean())
        df["Decayed_Rolling_AwayGoals"] = df.groupby('Away')['AwayGoals'].transform(lambda x: x.ewm(alpha=0.9).mean())
        df["Home_Advantage"] = df.groupby('Home')['HomeGoals'].transform(lambda x: x.rolling(5, min_periods=1).mean()) - df.groupby('Away')['AwayGoals'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        
        # Calculate streaks for old data
        df = calculate_streaks(df)
    
    elif dataset_type == 'new':
        df["Venue_Code"] = df["Venue"].astype("category").cat.codes
        df["Team_Code"] = df["Team"].astype("category").cat.codes
        df["Opponent_Code"] = df["Opponent"].astype("category").cat.codes
        df["Day_Code"] = pd.to_datetime(df["Date"]).dt.dayofweek
        df["Rolling_GF"] = df.groupby("Team")["GF"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df["Rolling_GA"] = df.groupby("Team")["GA"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df["Venue_Opp_Interaction"] = df["Venue_Code"] * df["Opponent_Code"]
        df["Decayed_Rolling_GF"] = df.groupby('Team')['GF'].transform(lambda x: x.ewm(alpha=0.9).mean())
        df["Decayed_Rolling_GA"] = df.groupby('Team')['GA'].transform(lambda x: x.ewm(alpha=0.9).mean())
        df["Team_Advantage"] = df.groupby('Team')['GF'].transform(lambda x: x.rolling(5, min_periods=1).mean()) - df.groupby('Opponent')['GA'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        
        # If old streaks are provided, use them to fill NaN values in the new data
        if old_streaks is not None:
            for team in df['Team'].unique():
                team_old_streaks = old_streaks.get(team, {})
                df.loc[df['Team'] == team, 'Home_Streak_Wins'] = df.loc[df['Team'] == team, 'Home_Streak_Wins'].fillna(method='ffill').fillna(team_old_streaks.get('Home_Streak_Wins', 0))
                df.loc[df['Team'] == team, 'Away_Streak_Losses'] = df.loc[df['Team'] == team, 'Away_Streak_Losses'].fillna(method='ffill').fillna(team_old_streaks.get('Away_Streak_Losses', 0))
    
    else:
        raise ValueError("Invalid dataset type provided. Use 'old' or 'new'.")
    
    return df

def extract_streaks(df):
    """Extract the last known streaks for each team from old data."""
    streaks = {}
    for team in df['Home'].unique():
        last_streak = df[df['Home'] == team].iloc[-1]
        streaks[team] = {
            'Home_Streak_Wins': last_streak['Home_Streak_Wins'],
            'Away_Streak_Losses': last_streak['Away_Streak_Losses']
        }
    return streaks

def apply_feature_engineering():
    # Load the preprocessed data
    old_matches = pd.read_csv('data/preprocessed_old_matches.csv')
    new_matches = pd.read_csv('data/preprocessed_new_matches.csv')

    # Add features to old matches and calculate streaks
    old_matches = add_features(old_matches, 'old')

    # Extract streaks from old matches
    old_streaks = extract_streaks(old_matches)

    # Add features to new matches using streak data from old matches
    new_matches = add_features(new_matches, 'new', old_streaks=old_streaks)

    # Rename new_matches columns to match old_matches
    new_matches.rename(columns={
        "Team_Code": "Venue_Code", 
        "Opponent_Code": "Opp_Code", 
        "Rolling_GF": "Rolling_HomeGoals", 
        "Rolling_GA": "Rolling_AwayGoals", 
        "Decayed_Rolling_GF": "Decayed_Rolling_HomeGoals", 
        "Decayed_Rolling_GA": "Decayed_Rolling_AwayGoals", 
        "Team_Advantage": "Home_Advantage"
    }, inplace=True)

    # Ensure both datasets have the same columns for modeling
    missing_cols = [col for col in old_matches.columns if col not in new_matches.columns]
    for col in missing_cols:
        new_matches[col] = np.nan

    # Align columns in the same order as old_matches
    new_matches = new_matches[old_matches.columns]

    # Save the feature-engineered data
    old_matches.to_csv('data/fe_old_matches.csv', index=False)
    new_matches.to_csv('data/fe_new_matches.csv', index=False)

    print("Feature engineering completed and files saved.")

if __name__ == "__main__":
    apply_feature_engineering()
