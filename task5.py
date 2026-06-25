import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Load Dataset
df = pd.read_csv("Housing.csv")

# Features
X = df[["area", "bedrooms", "bathrooms", "stories"]]

# Target
y = df["price"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -------------------
# Baseline
# -------------------

baseline_prediction = np.full(
    len(y_test),
    y_train.mean()
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_prediction
)

# -------------------
# ML Model
# -------------------

model = DecisionTreeRegressor(
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

model_mae = mean_absolute_error(
    y_test,
    predictions
)

# -------------------
# Results
# -------------------

print("Baseline MAE =", baseline_mae)
print("Model MAE =", model_mae)

if model_mae < baseline_mae:
    print("✅ Model beats Baseline")
else:
    print("❌ Model does not beat Baseline")

results = pd.DataFrame()

results["Actual Price"] = y_test.values
results["Predicted Price"] = predictions

results["Error"] = abs(
    results["Actual Price"] -
    results["Predicted Price"]
)

print("\nTop 10 Largest Errors:")
print(
    results.sort_values(
        by="Error",
        ascending=False
    ).head(10)
)    