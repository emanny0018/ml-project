import pandas as pd

# Load the datasets
old_matches_path = '/Users/ericiyen/mlapp/new-mlapp/premier-league-matches.csv'
new_matches_path = '/Users/ericiyen/mlapp/new-mlapp/matches-2023-2024.csv'

old_matches = pd.read_csv(old_matches_path)
new_matches = pd.read_csv(new_matches_path)

# Check columns in each dataset
print("Columns in old_matches:", old_matches.columns)
print("Columns in new_matches:", new_matches.columns)

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

# Display a sample to confirm the mapping
sample_size = 10
sample = new_matches[['Date', 'Venue', 'Opponent', 'Result', 'Target']].sample(sample_size)
print("Sample of Mapped Data:")
print(sample)

# Validation complete, save the mapped data
mapped_data_path = '/Users/ericiyen/mlapp/new-mlapp/mapped_matches_2023_2024.csv'
new_matches.to_csv(mapped_data_path, index=False)
print(f"Mapped data saved to {mapped_data_path}")
