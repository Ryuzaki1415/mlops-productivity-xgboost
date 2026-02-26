import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from config import DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, MODEL_PATH
from feature_engineering import create_features


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Creating features...")

    df = create_features(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    print("Xtrain Columns")
    print(X_train.columns)
    print("Loading trained model...")

    model = joblib.load(MODEL_PATH)
    print("Making predictions...")

    preds = model.predict(X_test)

    print("\nFirst 10 Predictions:")
    print(preds[:10])

    print("\nFirst 10 Actual Values:")
    print(y_test.iloc[:10].values)

    print("\nMean prediction:", preds.mean())
    print("Mean actual:", y_test.mean())


if __name__ == "__main__":
    main()