import pytest
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score


@pytest.fixture
def trained_model():
    # Load the trained model from the file
    model = joblib.load('model.pkl')
    return model


@pytest.fixture
def test_data():
    # Load test data
    data = pd.read_csv('dataset.csv')
    df = pd.DataFrame(data)
    X_test = df.drop(columns=['Product_id', 'Price', 'Sale'])
    y_test = df['Price']
    return X_test, y_test


def test_mean_squared_error(trained_model, test_data):
    model = trained_model
    X_test, y_test = test_data
    # Make predictions
    y_pred = model.predict(X_test)
    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)
    # Check if mse is not None and greater than or equal to 0
    assert mse is not None
    assert mse >= 0


def test_r_squared_score(trained_model, test_data):
    model = trained_model
    X_test, y_test = test_data
    # Make predictions
    y_pred = model.predict(X_test)
    # Calculate R-squared score
    r2 = r2_score(y_test, y_pred)
    # Check if r2 is not None and between 0 and 1
    assert r2 is not None
    assert 0 <= r2 <= 1
