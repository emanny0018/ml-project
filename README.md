# Premier League Match Prediction

This project aims to predict the outcomes of Premier League matches using advanced machine learning techniques. The project is structured to automate data preprocessing, feature engineering, model training, and evaluation using GitHub Actions.

## Project Structure

- `data/`: Contains raw, cleaned, and engineered data files.
- `scripts/`: Contains Python scripts for data processing, feature engineering, model training, and evaluation.
- `models/`: Contains the saved trained model.
- `.github/workflows/`: Contains the GitHub Actions workflow file.
- `requirements.txt`: Lists all Python dependencies.
- `README.md`: Project documentation.

## How to Run

1. **Preprocess Data**: Clean and preprocess the data by running `scripts/data_preprocessing.py`.
2. **Engineer Features**: Create new features from the cleaned data using `scripts/feature_engineering.py`.
3. **Train the Model**: Train the machine learning model using `scripts/model_training.py`.
4. **Evaluate the Model**: Evaluate the trained model's performance using `scripts/model_evaluation.py`.
5. **Predict Matches**: Predict the outcome of future matches using `scripts/predict_match.py`.

## Automated Pipeline

The project is fully automated using GitHub Actions. Every time you push changes to the repository, the pipeline will:
- Preprocess the data
- Engineer features
- Train the model
- Evaluate the model

Check the GitHub Actions tab in your repository to view the results of the pipeline.

## Dependencies

Install the necessary dependencies with:
```bash
pip install -r requirements.txt

