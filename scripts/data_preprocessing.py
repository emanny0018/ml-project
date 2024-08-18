import pandas as pd

def map_ftr_to_target(df):
    if 'FTR' in df.columns:
        df['Target'] = df['FTR'].map({'H': 0, 'A': 1, 'D': 2})
    elif 'Result' in df.columns:
        df['Target'] = df['Result'].map({'W': 0, 'L': 1, 'D': 2})
    else:
        raise ValueError("Neither 'FTR' nor 'Result' columns found in the dataset.")
    return df

def load_and_preprocess_data():
    # Load datasets
    old_matches = pd.read_csv('/Users/ericiyen/mlapp/new-mlapp/premier-league-matches.csv')
    new_matches = pd.read_csv('/Users/ericiyen/mlapp/new-mlapp/mapped_matches_2023_2024.csv')

    # Map FTR/Result to Target
    old_matches = map_ftr_to_target(old_matches)
    new_matches = map_ftr_to_target(new_matches)

    # Save preprocessed data
    old_matches.to_csv('/Users/ericiyen/mlapp/new-mlapp/preprocessed_old_matches.csv', index=False)
    new_matches.to_csv('/Users/ericiyen/mlapp/new-mlapp/preprocessed_new_matches.csv', index=False)

    return old_matches, new_matches

if __name__ == "__main__":
    old_matches, new_matches = load_and_preprocess_data()
    print("Data preprocessing completed.")

    # Display the first few rows for verification
    print("\nFirst few rows of old_matches with 'Target' column:")
    print(old_matches[['FTR', 'Target']].head(20))

    print("\nFirst few rows of new_matches with 'Target' column:")
    print(new_matches[['Result', 'Target']].head(20))
