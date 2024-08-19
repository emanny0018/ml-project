import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load feature-engineered datasets
train = pd.read_csv('data/fe_old_matches.csv')
test = pd.read_csv('data/fe_new_matches.csv')

# Define predictors
predictors = [
    "Venue_Code", "Opp_Code", "Day_Code", "Rolling_HomeGoals", 
    "Rolling_AwayGoals", "Venue_Opp_Interaction", 
    "Decayed_Rolling_HomeGoals", "Decayed_Rolling_AwayGoals", 
    "Home_Advantage"
]

# Split train and validation sets from the old matches dataset
X_train, X_val, y_train, y_val = train_test_split(
    train[predictors], train["Target"], test_size=0.2, random_state=42
)

# Initialize the models
xgb = XGBClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)

# Combine them in a VotingClassifier
voting_clf = VotingClassifier(
    estimators=[('xgb', xgb), ('rf', rf)],
    voting='soft'
)

# Train the model
voting_clf.fit(X_train, y_train)

# Validate the model
y_pred = voting_clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.2%}")

# Save the trained model
import joblib
joblib.dump(voting_clf, 'models/voting_classifier.pkl')
