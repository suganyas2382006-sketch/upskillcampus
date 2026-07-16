import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
data = pd.read_csv("dataset/traffic_data.csv")

# Features and Target
X = data[["Hour", "Day", "Temperature", "Rainfall"]]
y = data["Traffic"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))

# Save model
joblib.dump(model, "traffic_model.pkl")
print("Model saved successfully!")