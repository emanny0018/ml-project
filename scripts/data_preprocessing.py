import pandas as pd

# Paths for input and output files
old_matches_path = 'data/premier-league-matches.csv'
new_matches_path = 'data/matches-2023-2024.csv'
mapped_data_path = 'data/mapped_matches_2023_2024.csv'

# Load the datasets
old_matches = pd.read_csv(old_matches_path)
new_matches = pd.read_csv(new_matches_path)

# Map the results to target values
def map_ftr_to_target(ftr):
    if ftr == 'H':
        return 0
    elif ftr == 'A':
        return 1
    else:
        return 2  # Draws

old_matches['Target'] = old_matches['FTR'].map(map_ftr_to_target)

def map_result_to_target(row):
    if row['Result'] == 'W':
        return 0 if row['Venue'] == "Home" else 1
    elif row['Result'] == 'L':
        return 1 if row['Venue'] == "Home" else 0
    else:
        return 2  # Draws

new_matches['Target'] = new_matches.apply(map_result_to_target, axis=1)

# Save the mapped data to the specified path
new_matches.to_csv(mapped_data_path, index=False)
print(f"Mapped data saved to {mapped_data_path}")
