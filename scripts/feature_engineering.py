import pandas as pd

def add_features(df):
    if 'Home' in df.columns and 'Away' in df.columns:
        df["Venue_Code"] = df["Home"].astype("category").cat.codes
        df["Opp_Code"] = df["Away"].astype("category").cat.codes
        df["Day_Code"] = pd.to_datetime(df["Date"]).dt.dayofweek
        df["Rolling_HomeGoals"] = df.groupby("Home")["HomeGoals"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df["Rolling_AwayGoals"] = df.groupby("Away")["AwayGoals"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df["Venue_Opp_Interaction"] = df["Venue_Code"] * df["Opp_Code"]
        df["Decayed_Rolling_HomeGoals"] = df.groupby('Home')['HomeGoals'].transform(lambda x: x.ewm(alpha=0.9).mean())
        df["Decayed_Rolling_AwayGoals"] = df.groupby('Away')['AwayGoals'].transform(lambda x: x.ewm(alpha=0.9).mean())
        df["Home_Advantage"] = df.groupby('Home')['HomeGoals'].transform(lambda x: x.rolling(5).mean()) - df.groupby('Away')['AwayGoals'].transform(lambda x: x.rolling(5).mean())
    else:
        print("The required columns 'Home' and 'Away' are missing in the DataFrame.")
        print(f"Available columns: {df.columns.tolist()}")
    return df

def apply_feature_engineering():
    # Load the preprocessed data
    old_matches = pd.read_csv('data/preprocessed_old_matches.csv')
    new_matches = pd.read_csv('data/preprocessed_new_matches.csv')

    # Print the column names to debug
    print("Columns in old_matches:")
    print(old_matches.columns.tolist())
    
    print("Columns in new_matches:")
    print(new_matches.columns.tolist())

    # Add features
    old_matches = add_features(old_matches)
    new_matches = add_features(new_matches)

    # Save the feature-engineered data
    old_matches.to_csv('data/fe_old_matches.csv', index=False)
    new_matches.to_csv('data/fe_new_matches.csv', index=False)

    print("Feature engineering completed and files saved.")

if __name__ == "__main__":
    apply_feature_engineering()
