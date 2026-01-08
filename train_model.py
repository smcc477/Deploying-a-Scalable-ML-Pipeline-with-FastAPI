import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    save_model,
    performance_on_categorical_slice,
)

DATA_PATH = "data/census.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
LB_PATH = os.path.join(MODEL_DIR, "lb.pkl")
SLICE_OUTPUT_PATH = "slice_output.txt"

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

LABEL = "salary"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    data = pd.read_csv(DATA_PATH)

    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data[LABEL]
    )

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
        training=True,
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)

    save_model(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    save_model(encoder, ENCODER_PATH)
    print(f"Model saved to {ENCODER_PATH}")

    save_model(lb, LB_PATH)
    print(f"Model saved to {LB_PATH}")

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}")

    with open(SLICE_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for feature in CATEGORICAL_FEATURES:
            for value in sorted(test[feature].dropna().unique()):
                p, r, fb = performance_on_categorical_slice(
                    data=test,
                    column_name=feature,
                    slice_value=value,
                    categorical_features=CATEGORICAL_FEATURES,
                    label=LABEL,
                    encoder=encoder,
                    lb=lb,
                    model=model,
                )
                count = int((test[feature] == value).sum())
                f.write(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}\n")
                f.write(f"{feature}: {value}, Count: {count}\n")

    print(f"Slice output saved to {SLICE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
