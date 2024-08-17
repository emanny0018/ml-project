import os
import pandas as pd
from scripts.feature_engineering import engineer_features, save_engineered_data

def test_engineer_features():
    # Create a sample DataFrame with minimal data
    df = pd.DataFrame({
        'Home': ['TeamA', 'TeamB'],
        'Away': ['TeamC', 'TeamD'],
        'HomeGoals': [2, 1],
        'AwayGoals': [1, 2]
    })
    df_engineered = engineer_features(df)
    # Test if new features were added
    assert 'Rolling_HomeGoals' in df_engineered.columns, "Rolling_HomeGoals feature not added"
    assert 'Rolling_AwayGoals' in df_engineered.columns, "Rolling_AwayGoals feature not added"

def test_save_engineered_data(tmpdir):
    # Create a sample DataFrame
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    # Define a temporary file path
    file_path = os.path.join(tmpdir, 'engineered_data.csv')
    # Save the DataFrame to the file
    save_engineered_data(df, file_path)
    # Test if the file was created
    assert os.path.exists(file_path), "The engineered data file was not saved"

