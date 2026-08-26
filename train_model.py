import os
import pickle
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "dataset/traffic_data.csv"
MODEL_PATH = "model/traffic_model.pkl"

df = pd.read_csv(DATA_PATH)
df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df["Hour"] = df["DateTime"].dt.hour
df["Day"] = df["DateTime"].dt.day
df["Month"] = df["DateTime"].dt.month
df["DayOfWeekNum"] = df["DateTime"].dt.dayofweek

features = ["Hour", "Day", "Month", "DayOfWeekNum", "Weather", "Holiday"]
target = "TrafficVolume"

X = df[features]
y = df[target]

categorical = ["Weather"]
numeric = ["Hour", "Day", "Month", "DayOfWeekNum", "Holiday"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", "passthrough", numeric)
])

model = RandomForestRegressor(
    n_estimators=250,
    max_depth=18,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

os.makedirs("model", exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)

print("\nMODEL TRAINED SUCCESSFULLY")
print("-" * 40)
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")
print(f"\nSaved model: {MODEL_PATH}")
