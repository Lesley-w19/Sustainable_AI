import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model():
    df = pd.read_csv("energy_training_data.csv")

    X = df.drop("energy_kwh", axis=1)
    y = df["energy_kwh"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )
    model.fit(X_train, y_train)

    print("Train R²:", model.score(X_train, y_train))
    print("Test R²:", model.score(X_test, y_test))

    joblib.dump(model, "energy_predictor.pkl")

if __name__ == "__main__":
    train_model()
