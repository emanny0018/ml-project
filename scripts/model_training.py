import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the feature-engineered datasets
train = pd.read_csv('data/fe_old_matches.csv')
test = pd.read_csv('data/fe_new_matches.csv')

# Define the predictors based on feature engineering
advanced_predictors = [
    "Venue_Code", "Opp_Code", "Day_Code", "Rolling_HomeGoals", "Rolling_AwayGoals", 
    "Venue_Opp_Interaction", "Decayed_Rolling_HomeGoals", "Decayed_Rolling_AwayGoals", 
    "Home_Advantage", "Home_Streak_Wins", "Away_Streak_Losses"
]

# Define a parameter grid for XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

# Initialize XGBoost classifier
xgb_classifier = XGBClassifier(random_state=42)

# Perform GridSearchCV with 3-fold cross-validation for XGBoost
grid_search_xgb = GridSearchCV(
    estimator=xgb_classifier,
    param_grid=param_grid_xgb,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

# Fit the model
grid_search_xgb.fit(train[advanced_predictors], train["Target"])
best_xgb_classifier = grid_search_xgb.best_estimator_

# Initialize RandomForest classifier
rf_classifier = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=7)

# Ensemble using Voting Classifier
voting_classifier = VotingClassifier(
    estimators=[('xgb', best_xgb_classifier), ('rf', rf_classifier)],
    voting='soft'
)

# Fit the ensemble model
voting_classifier.fit(train[advanced_predictors], train["Target"])

# Make predictions on the test set
test_predictions = voting_classifier.predict(test[advanced_predictors])

# Calculate accuracy
accuracy = accuracy_score(test["Target"], test_predictions)
accuracy_percentage = accuracy * 100

# Output results
print(f"Accuracy: {accuracy_percentage:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(test["Target"], test_predictions))
print("Classification Report:")
print(classification_report(test["Target"], test_predictions))

# Optionally, save the model (uncomment if needed)
# import joblib
# joblib.dump(voting_classifier, 'models/voting_classifier.pkl')
