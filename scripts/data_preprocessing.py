import os
import pandas as pd

def map_result_to_target(df):
    if 'Result' in df.columns:
        df['Target'] = df['Result'].map({'W': 0, 'L': 1, 'D': 2})
    else:
        print("'Result' column is missing. 'Target' column will not be created.")
    return df

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

    # Ensure the 'data' directory exists
    os.makedirs('data', exist_ok=True)

    # Save preprocessed data
    old_matches_output_path = 'data/preprocessed_old_matches.csv'
    new_matches_output_path = 'data/preprocessed_new_matches.csv'
    
    old_matches.to_csv(old_matches_output_path, index=False)
    new_matches.to_csv(new_matches_output_path, index=False)

    # Check if files were saved correctly
    if os.path.isfile(old_matches_output_path) and os.path.isfile(new_matches_output_path):
        print("Preprocessed data saved successfully.")
    else:
        print("Error: Preprocessed data files were not saved.")

    # List the contents of the data directory for confirmation
    print("\nContents of 'data' directory after saving:")
    for file_name in os.listdir('data'):
        print(file_name)

    return old_matches, new_matches

if __name__ == "__main__":
    old_matches, new_matches = load_and_preprocess_data()

    # Display the first few rows for verification
    print("\nFirst few rows of old_matches with 'Target' column:")
    print(old_matches[['FTR', 'Target']].head(20))

    print("\nFirst few rows of new_matches with 'Target' column:")
    print(new_matches[['Result', 'Target']].head(20))
