import pandas as pd

def load_cleaned_data(file_path):
    """Load cleaned dataset."""
    return pd.read_csv(file_path)

def engineer_features(df):
    """Engineer new features for the dataset."""

    # Check if 'Venue_Code' and 'Opp_Code' exist, and create them if they don't
    if 'Venue_Code' not in df.columns:
        df['Venue_Code'] = df['Venue'].astype('category').cat.codes
    if 'Opp_Code' not in df.columns:
        df['Opp_Code'] = df['Opponent'].astype('category').cat.codes

    # Example feature engineering steps
    df['Rolling_HomeGoals'] = df.groupby('Home')['HomeGoals'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
    df['Rolling_AwayGoals'] = df.groupby('Away')['AwayGoals'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
    
    df['Venue_Opp_Interaction'] = df['Venue_Code'] * df['Opp_Code']
    df['Recent_Form_Home'] = df.groupby('Home')['Target'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    df['Recent_Form_Away'] = df.groupby('Away')['Target'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    
    df['Decayed_Rolling_HomeGoals'] = df.groupby('Home')['HomeGoals'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    df['Decayed_Rolling_AwayGoals'] = df.groupby('Away')['AwayGoals'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    
    df['Home_Advantage'] = (df['Venue'] == 'Home').astype(int)
    
    return df

def save_engineered_data(df, output_path):
    """Save the dataset with engineered features."""
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Load cleaned data
    df1_clean = load_cleaned_data('data/premier-league-matches-cleaned.csv')
    df2_clean = load_cleaned_data('data/matches-2023-2024-cleaned.csv')

    # Engineer features
    df1_engineered = engineer_features(df1_clean)
    df2_engineered = engineer_features(df2_clean)

    # Save engineered data
    save_engineered_data(df1_engineered, 'data/premier-league-matches-engineered.csv')
    save_engineered_data(df2_engineered, 'data/matches-2023-2024-engineered.csv')
