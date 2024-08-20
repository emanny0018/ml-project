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
        
        # Ensure streak columns exist in new data
        df['Home_Streak_Wins'] = np.nan
        df['Away_Streak_Losses'] = np.nan
        
        if old_streaks is not None:
            for team, streaks in old_streaks.items():
                team_old_streaks = old_streaks.get(team, {})
                df.loc[df['Team'] == team, 'Home_Streak_Wins'] = df.loc[df['Team'] == team, 'Home_Streak_Wins'].fillna(method='ffill').fillna(team_old_streaks.get('Home_Streak_Wins', 0))
                df.loc[df['Team'] == team, 'Away_Streak_Losses'] = df.loc[df['Team'] == team, 'Away_Streak_Losses'].fillna(method='ffill').fillna(team_old_streaks.get('Away_Streak_Losses', 0))
    
    else:
        raise ValueError("Invalid dataset type provided. Use 'old' or 'new'.")

    return df

def apply_feature_engineering():
    # Load the preprocessed data
    old_matches = pd.read_csv('data/preprocessed_old_matches.csv')
    new_matches = pd.read_csv('data/preprocessed_new_matches.csv')

    # Add features to old matches
    old_matches = add_features(old_matches, 'old')

    # Store streaks from old matches
    old_streaks = old_matches.groupby('Home').agg({
        'Home_Streak_Wins': 'last',
        'Away_Streak_Losses': 'last'
    }).to_dict('index')

    # Add features to new matches
    new_matches = add_features(new_matches, 'new', old_streaks=old_streaks)

    # Save the feature-engineered data
    old_matches.to_csv('data/fe_old_matches.csv', index=False)
    new_matches.to_csv('data/fe_new_matches.csv', index=False)

    print("Feature engineering completed and files saved.")

    # Display the first few rows for verification
    print("\nFirst few rows of new_matches after feature engineering:")
    print(new_matches.head(20))

if __name__ == "__main__":
    apply_feature_engineering()
