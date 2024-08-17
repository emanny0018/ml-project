import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the datasets
train = pd.read_csv('data/premier-league-matches-engineered.csv')
test = pd.read_csv('data/matches-2023-2024-engineered.csv')

# Define predictors
advanced_predictors = [
    "Venue_Code", "Opp_Code", "Day_Code", "Rolling_HomeGoals", "Rolling_AwayGoals", 
    "Venue_Opp_Interaction", "Recent_Form_Home", "Recent_Form_Away", 
    "Decayed_Rolling_HomeGoals", "Decayed_Rolling_AwayGoals", "Home_Advantage"
]

# Add new features
train['Home_Streak_Wins'] = train.groupby('Home')['Target'].transform(lambda x: (x == 0).rolling(window=5, min_periods=1).sum())
train['Away_Streak_Losses'] = train.groupby('Away')['Target'].transform(lambda x: (x == 1).rolling(window=5, min_periods=1).sum())
test['Home_Streak_Wins'] = test.groupby('Home')['Target'].transform(lambda x: (x == 0).rolling(window=5, min_periods=1).sum())
test['Away_Streak_Losses'] = test.groupby('Away')['Target'].transform(lambda x: (x == 1).rolling(window=5, min_periods=1).sum())

advanced_predictors += ['Home_Streak_Wins', 'Away_Streak_Losses']

# Define class weights based on current class distribution
class_weights_dict = {0: 1, 1: 2, 2: 3}

# Define a parameter grid for XGBoost
param_dist = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

# Initialize XGBoost classifier
xgb_classifier = XGBClassifier(random_state=42, scale_pos_weight=class_weights_dict)

# Perform RandomizedSearchCV with 3-fold cross-validation
random_search_xgb = RandomizedSearchCV(
    estimator=xgb_classifier,
    param_distributions=param_dist,
    n_iter=10,
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit the model
random_search_xgb.fit(train[advanced_predictors], train["Target"])
best_xgb_classifier = random_search_xgb.best_estimator_

# Make predictions on the test set
test_predictions = best_xgb_classifier.predict(test[advanced_predictors])

# Calculate accuracy
accuracy = accuracy_score(test["Target"], test_predictions)
accuracy_percentage = accuracy * 100
print(f"Accuracy: {accuracy_percentage:.2f}%")

# Print confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(test["Target"], test_predictions))
print("Classification Report:")
print(classification_report(test["Target"], test_predictions))
