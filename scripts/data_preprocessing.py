import pandas as pd

# Define relative paths for the datasets
old_matches_path = 'data/premier-league-matches.csv'
new_matches_path = 'data/matches-2023-2024.csv'

# Load the datasets
old_matches = pd.read_csv(old_matches_path)
new_matches = pd.read_csv(new_matches_path)

# Step 1: Create the "Target" column in both datasets
def map_ftr_to_target(ftr):
    if ftr == 'H':
        return 0
    elif ftr == 'A':
        return 1
    else:
        return 2  # For draws

old_matches['Target'] = old_matches['FTR'].map(map_ftr_to_target)

def map_result_to_target(row):
    if row['Result'] == 'W':
        return 0 if row['Venue'] == "Home" else 1
    elif row['Result'] == 'L':
        return 1 if row['Venue'] == "Home" else 0
    else:
        return 2  # For draws

new_matches['Target'] = new_matches.apply(map_result_to_target, axis=1)

# Save the mapped data to CSV files with expected filenames
old_cleaned_data_path = 'data/premier-league-matches-cleaned.csv'
new_cleaned_data_path = 'data/matches-2023-2024-cleaned.csv'
old_matches.to_csv(old_cleaned_data_path, index=False)
new_matches.to_csv(new_cleaned_data_path, index=False)

print(f"Old matches cleaned data saved to {old_cleaned_data_path}")
print(f"New matches cleaned data saved to {new_cleaned_data_path}")
