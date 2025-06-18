import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("sample_cgpa_dataset.csv")

# Display sample rows
print("Sample data:")
print(df.head())

# Features and target
X = df[["cumulative_obtained", "cumulative_total", "cumulative_credits", "marks_per_credit", "percentage"]]
y = df["final_cgpa"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"✔️ RMSE: {rmse:.2f}")
print(f"✔️ R² Score: {r2:.2f}")

# Save model in same folder
joblib.dump(model, "cgpa_model.pkl")
print("✅ Model saved as cgpa_model.pkl in the same folder.")
