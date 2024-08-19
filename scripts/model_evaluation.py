import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import load

# Load the test data and trained model
test = pd.read_csv('data/fe_new_matches.csv')
model = load('data/voting_classifier.pkl')

# Define predictors based on your feature engineering process
advanced_predictors = [
    "Venue_Code", "Opp_Code", "Day_Code", "Rolling_HomeGoals", "Rolling_AwayGoals", 
    "Venue_Opp_Interaction", "Decayed_Rolling_HomeGoals", "Decayed_Rolling_AwayGoals", 
    "Home_Advantage", "Home_Streak_Wins", "Away_Streak_Losses"
]

# Make predictions on the test set
test_predictions = model.predict(test[advanced_predictors])

# Calculate accuracy
accuracy = accuracy_score(test["Target"], test_predictions)
accuracy_percentage = accuracy * 100

# Output results
print(f"Accuracy: {accuracy_percentage:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(test["Target"], test_predictions))
print("Classification Report:")
print(classification_report(test["Target"], test_predictions))
