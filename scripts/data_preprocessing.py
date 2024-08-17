import pandas as pd

def load_data(file_path):
    """Load dataset from a file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the dataset."""
    df.dropna(inplace=True)  # Drop missing values
    df.drop_duplicates(inplace=True)  # Remove duplicate rows
    return df

def save_cleaned_data(df, output_path):
    """Save the cleaned dataset."""
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Load data
    df1 = load_data('data/premier-league-matches.csv')
    df2 = load_data('data/matches-2023-2024.csv')

    # Clean data
    df1_clean = clean_data(df1)
    df2_clean = clean_data(df2)

    # Save cleaned data
    save_cleaned_data(df1_clean, 'data/premier-league-matches-cleaned.csv')
    save_cleaned_data(df2_clean, 'data/matches-2023-2024-cleaned.csv')

