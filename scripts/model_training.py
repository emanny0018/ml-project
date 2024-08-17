import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from joblib import dump

# Load the preprocessed data
train = pd.read_csv('data/premier-league-matches-engineered.csv')

# Define advanced predictors
advanced_predictors = [
    "Venue_Code", "Opp_Code", "Day_Code", "Rolling_HomeGoals", "Rolling_AwayGoals", 
    "Venue_Opp_Interaction", "Recent_Form_Home", "Recent_Form_Away", 
    "Decayed_Rolling_HomeGoals", "Decayed_Rolling_AwayGoals", "Home_Advantage"
]

# Adding new features: Streak of wins for Home team and streak of losses for Away team
train['Home_Streak_Wins'] = train.groupby('Home')['Target'].transform(lambda x: (x == 0).rolling(window=5, min_periods=1).sum())
train['Away_Streak_Losses'] = train.groupby('Away')['Target'].transform(lambda x: (x == 1).rolling(window=5, min_periods=1).sum())

# Add the new features to the predictors
advanced_predictors += ['Home_Streak_Wins', 'Away_Streak_Losses']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train[advanced_predictors], train["Target"], test_size=0.2, random_state=42)

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
grid_search_xgb.fit(X_train, y_train)
best_xgb_classifier = grid_search_xgb.best_estimator_

# Initialize RandomForest classifier
rf_classifier = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=7)

# Ensemble using Voting Classifier
voting_classifier = VotingClassifier(
    estimators=[('xgb', best_xgb_classifier), ('rf', rf_classifier)],
    voting='soft'
)

# Fit the ensemble model
voting_classifier.fit(X_train, y_train)

# Save the trained model
dump(voting_classifier, 'models/model.pkl')

