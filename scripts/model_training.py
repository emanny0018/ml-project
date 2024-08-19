import pandas as pd
from sklearn.ensemble import VotingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Define the predictors (adjust these based on your feature engineering)
advanced_predictors = [
    "Venue_Code", "Opp_Code", "Day_Code", "Rolling_HomeGoals", "Rolling_AwayGoals", 
    "Venue_Opp_Interaction", "Decayed_Rolling_HomeGoals", "Decayed_Rolling_AwayGoals", 
    "Home_Advantage"
]

def train_models(train, test):
    # Define a column transformer to handle missing values
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), advanced_predictors)
        ]
    )

    # Initialize models with pipelines that include the preprocessor
    xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', XGBClassifier(random_state=42))])
    
    hist_gb_pipeline = Pipeline(steps=[('classifier', HistGradientBoostingClassifier(random_state=42))])

    # Ensemble using Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[('xgb', xgb_pipeline), ('hist_gb', hist_gb_pipeline)],
        voting='soft'
    )

    # Train the ensemble model
    voting_clf.fit(train[advanced_predictors], train["Target"])

    # Predict on test data
    predictions = voting_clf.predict(test[advanced_predictors])

    # Evaluate the model
    accuracy = accuracy_score(test["Target"], predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix(test["Target"], predictions))
    print("Classification Report:")
    print(classification_report(test["Target"], predictions))

    return voting_clf

if __name__ == "__main__":
    # Load the feature-engineered datasets
    train = pd.read_csv('data/fe_old_matches.csv')
    test = pd.read_csv('data/fe_new_matches.csv')

    # Train the models and evaluate
    voting_clf = train_models(train, test)
