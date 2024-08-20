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

    # Convert team names to lowercase
    old_matches['Home'] = old_matches['Home'].str.lower()
    old_matches['Away'] = old_matches['Away'].str.lower()
    new_matches['Team'] = new_matches['Team'].str.lower()
    new_matches['Opponent'] = new_matches['Opponent'].str.lower()

    # Apply the mapping functions
    old_matches = map_ftr_to_target(old_matches)
    new_matches = map_result_to_target(new_matches)

    # Rename columns in new matches to match the old dataset
    new_matches_renamed = new_matches.rename(columns={
        'Team': 'Home', 'Opponent': 'Away', 'GF': 'HomeGoals', 'GA': 'AwayGoals'})

    # Combine old and new matches for streak calculation
    combined_matches = pd.concat([old_matches, new_matches_renamed], ignore_index=True)

    # Sort by date to ensure continuity
    combined_matches.sort_values('Date', inplace=True)
    
    # Reset the index after sorting
    combined_matches.reset_index(drop=True, inplace=True)

    # Save preprocessed data
    combined_output_path = 'data/combined_matches.csv'
    old_matches_output_path = 'data/preprocessed_old_matches.csv'
    new_matches_output_path = 'data/preprocessed_new_matches.csv'
    
    combined_matches.to_csv(combined_output_path, index=False)
    old_matches.to_csv(old_matches_output_path, index=False)
    new_matches.to_csv(new_matches_output_path, index=False)

    # Check if files were saved correctly
    if os.path.isfile(combined_output_path) and os.path.isfile(old_matches_output_path) and os.path.isfile(new_matches_output_path):
        print("Preprocessed data saved successfully.")
    else:
        print("Error: Preprocessed data files were not saved.")

    # List the contents of the data directory for confirmation
    print("\nContents of 'data' directory after saving:")
    for file_name in os.listdir('data'):
        print(file_name)

    return combined_matches, old_matches, new_matches

if __name__ == "__main__":
    combined_matches, old_matches, new_matches = load_and_preprocess_data()

    # Display the first few rows for verification
    print("\nFirst few rows of combined_matches with 'Target' column:")
    print(combined_matches[['Date', 'Home', 'Away', 'Target']].head(20))

    print("\nFirst few rows of new_matches with 'Target' column:")
    print(new_matches[['Result', 'Target']].head(20))
