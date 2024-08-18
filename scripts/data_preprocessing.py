import pandas as pd

# Function to map 'Result' in new_matches to 'Target'
def map_result_to_target(df):
    if 'Result' in df.columns:
        df['Target'] = df['Result'].map({'W': 0, 'L': 1, 'D': 2})
    else:
        print("'Result' column is missing. 'Target' column will not be created.")
    return df

# Function to map 'FTR' in old_matches to 'Target'
def map_ftr_to_target(df):
    if 'FTR' in df.columns:
        df['Target'] = df['FTR'].map({'H': 0, 'A': 1, 'D': 2})
    else:
        print("'FTR' column is missing. 'Target' column will not be created.")
    return df

def load_and_preprocess_data():
    # Load datasets
    old_matches = pd.read_csv('data/premier-league-matches.csv')
    new_matches = pd.read_csv('data/mapped_matches_2023_2024.csv')

    # Apply the mapping functions
    old_matches = map_ftr_to_target(old_matches)
    new_matches = map_result_to_target(new_matches)

    # Save preprocessed data
    old_matches.to_csv('data/preprocessed_old_matches.csv', index=False)
    new_matches.to_csv('data/preprocessed_new_matches.csv', index=False)

    return old_matches, new_matches

if __name__ == "__main__":
    old_matches, new_matches = load_and_preprocess_data()
    print("Data preprocessing completed.")

    # Display the first few rows for verification
    print("\nFirst few rows of old_matches with 'Target' column:")
    print(old_matches[['FTR', 'Target']].head(20))

    print("\nFirst few rows of new_matches with 'Target' column:")
    print(new_matches[['Result', 'Target']].head(20))
