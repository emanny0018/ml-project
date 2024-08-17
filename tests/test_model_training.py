import os
from joblib import load
from scripts.model_training import voting_classifier, X_train, y_train

def test_model_training():
    # Test if the model is trained and not None
    assert voting_classifier is not None, "Model training failed, the classifier is None"
    # Test if the model can make predictions on the training data
    predictions = voting_classifier.predict(X_train)
    assert len(predictions) == len(y_train), "Prediction length does not match the training data length"

def test_save_model(tmpdir):
    # Save the trained model to a temporary directory
    model_path = os.path.join(tmpdir, 'model.pkl')
    from joblib import dump
    dump(voting_classifier, model_path)
    # Test if the model file was created
    assert os.path.exists(model_path), "The trained model was not saved"

