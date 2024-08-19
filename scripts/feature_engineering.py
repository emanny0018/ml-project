import pandas as pd
import numpy as np

def add_features(df, dataset_type):
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
        df['Home_Streak_Wins'] = df.groupby('Home')['Target'].transform(lambda x: (x == 0).rolling(window=5, min_periods=1).sum())
        df['Away_Streak_Losses'] = df.groupby('Away')['Target'].transform(lambda x: (x == 1).rolling(window=5, min_periods=1).sum())
    
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
    
    else:
        raise ValueError("Invalid dataset type provided. Use 'old' or 'new'.")

    return df

def apply_feature_engineering():
    # Load the preprocessed data
    old_matches = pd.read_csv('data/preprocessed_old_matches.csv')
    new_matches = pd.read_csv('data/preprocessed_new_matches.csv')

    # Add features
    old_matches = add_features(old_matches, 'old')
    new_matches = add_features(new_matches, 'new')

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
