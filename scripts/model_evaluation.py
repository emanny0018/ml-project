import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from joblib import load

# Load the test data and trained model
data = pd.read_csv('data/fe_combined_matches.csv')
model = load('data/voting_classifier.pkl')

# Split data into train and test sets
_, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data["Target"])

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

# Save results to a CSV file
results_df = pd.DataFrame({
    "Accuracy": [accuracy_percentage],
    "Confusion_Matrix": [conf_matrix.tolist()],
    "Classification_Report": [class_report]
})
results_df.to_csv('data/evaluation_results.csv', index=False)

print("Evaluation results saved to 'data/evaluation_results.csv'")

# Output results
print(f"Accuracy: {accuracy_percentage:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(test["Target"], test_predictions))
print("Classification Report:")
print(classification_report(test["Target"], test_predictions))
