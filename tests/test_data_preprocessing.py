import os
import pandas as pd
from scripts.data_preprocessing import load_data, clean_data, save_cleaned_data

def test_clean_data():
    # Create a sample DataFrame with duplicates and missing values
    df = pd.DataFrame({
        'A': [1, 2, 2, None],
        'B': [4, 5, 6, 6]
    })
    df_cleaned = clean_data(df)
    # Test if there are no missing values
    assert df_cleaned.isna().sum().sum() == 0, "There are still missing values after cleaning"
    # Test if there are no duplicate rows
    assert df_cleaned.duplicated().sum() == 0, "There are still duplicate rows after cleaning"

def test_save_cleaned_data(tmpdir):
    # Create a sample DataFrame
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    # Define a temporary file path
    file_path = os.path.join(tmpdir, 'cleaned_data.csv')
    # Save the DataFrame to the file
    save_cleaned_data(df, file_path)
    # Test if the file was created
    assert os.path.exists(file_path), "The cleaned data file was not saved"

def test_load_data():
    # Load the raw data
    df = load_data('data/premier-league-matches.csv')
    # Test if the DataFrame is not empty
    assert not df.empty, "The data file was not loaded correctly"

