import pandas as pd

def load_cleaned_data(file_path):
    """Load cleaned dataset."""
    return pd.read_csv(file_path)

def engineer_features(df, is_old_data=True):
    """Engineer new features for the dataset."""
    # (Feature engineering logic here...)
    return df

def save_engineered_data(df, output_path):
    """Save the dataset with engineered features."""
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Load the cleaned data using the correct filenames
    df1_clean = load_cleaned_data('data/mapped_matches_2023_2024.csv')
    df2_clean = load_cleaned_data('data/matches-2023-2024-cleaned.csv')

    # Engineer features
    df1_engineered = engineer_features(df1_clean, is_old_data=True)
    df2_engineered = engineer_features(df2_clean, is_old_data=False)

    # Save engineered data
    save_engineered_data(df1_engineered, 'data/premier-league-matches-engineered.csv')
    save_engineered_data(df2_engineered, 'data/matches-2023-2024-engineered.csv')
