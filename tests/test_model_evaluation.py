import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score
from scripts.model_evaluation import test, model

def test_model_loading():
    # Test if the model loads correctly
    loaded_model = load('models/model.pkl')
    assert loaded_model is not None, "Model could not be loaded"

def test_model_evaluation():
    # Test if the model makes predictions on the test set
    predictions = model.predict(test[model.get_params()['X_train'].columns])
    accuracy = accuracy_score(test["Target"], predictions)
    # Test if accuracy is within an expected range (e.g., 0.0 - 1.0)
    assert 0.0 <= accuracy <= 1.0, "Accuracy is out of expected bounds"

