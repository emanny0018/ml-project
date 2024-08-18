import pandas as pd

def load_data():
    # Load preprocessed datasets
    old_matches = pd.read_csv('data/preprocessed_old_matches.csv')
    new_matches = pd.read_csv('data/preprocessed_new_matches.csv')
    return old_matches, new_matches

def add_features(df):
    df["Venue_Code"] = df["Home"].astype("category").cat.codes
    df["Opp_Code"] = df["Away"].astype("category").cat.codes
    df["Day_Code"] = pd.to_datetime(df["Date"]).dt.dayofweek
    df["Rolling_HomeGoals"] = df.groupby("Home")["HomeGoals"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["Rolling_AwayGoals"] = df.groupby("Away")["AwayGoals"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["Venue_Opp_Interaction"] = df["Venue_Code"] * df["Opp_Code"]
    df["Decayed_Rolling_HomeGoals"] = df.groupby('Home')['HomeGoals'].transform(lambda x: x.ewm(alpha=0.9).mean())
    df["Decayed_Rolling_AwayGoals"] = df.groupby('Away')['AwayGoals'].transform(lambda x: x.ewm(alpha=0.9).mean())
    df["Home_Advantage"] = df.groupby('Home')['HomeGoals'].transform(lambda x: x.rolling(5).mean()) - df.groupby('Away')['AwayGoals'].transform(lambda x: x.rolling(5).mean())
    
    return df

def apply_feature_engineering():
    # Load the data
    old_matches, new_matches = load_data()

    # Add features
    old_matches = add_features(old_matches)
    new_matches = add_features(new_matches)

    # Save feature-engineered data
    old_matches.to_csv('data/fe_old_matches.csv', index=False)
    new_matches.to_csv('data/fe_new_matches.csv', index=False)

    print("Feature engineering completed successfully.")

if __name__ == "__main__":
    apply_feature_engineering()
