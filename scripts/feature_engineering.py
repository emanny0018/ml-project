import pandas as pd
import numpy as np

def add_features(df):
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

    return df

def apply_feature_engineering():
    # Load the preprocessed data
    combined_matches = pd.read_csv('data/combined_matches.csv')

    # Add features
    combined_matches = add_features(combined_matches)

    # Save the feature-engineered data
    combined_matches.to_csv('data/fe_combined_matches.csv', index=False)

    print("Feature engineering completed and files saved.")
    print("\nFirst few rows of combined_matches after feature engineering:")
    print(combined_matches.head(20))

if __name__ == "__main__":
    apply_feature_engineering()
