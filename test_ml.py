import pytest
import pandas as pd
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
# TODO: add necessary import


# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    # add description for the first test
    """
    data = pd.read_csv("data/census.csv").head(200)

    X, y, encoder, lb = process_data(
        data,
        categorical_features=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ],
        label="salary",
        training=True,
    )

    assert X is not None
    assert y is not None
    assert X.shape[0] == y.shape[0]


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # add description for the second test
    """
    data = pd.read_csv("data/census.csv").head(300)

    X, y, encoder, lb = process_data(
        data,
        categorical_features=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ],
        label="salary",
        training=True,
    )

    model = train_model(X, y)
    preds = inference(model, X)

    assert len(preds) == X.shape[0]
    assert set(pd.Series(preds).unique()).issubset({0, 1})


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    y_true = [0, 1, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 0, 1, 1]

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0
