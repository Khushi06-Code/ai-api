import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# load data
df = pd.read_csv("sepsis.csv")

# features (same jo API me use kar rahe ho)
X = df[[
    "hr_mean",
    "temp_celsius_mean",
    "spo2_mean",
    "sbp_mean",
    "dbp_mean",
    "respiratory_rate_mean",
    "wbc",
    "glucose"
]]
y = df["sepsis_label"]

# handle missing values
X = X.fillna(0)

# 🔥 split data (IMPORTANT for accuracy)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 🔥 prediction for accuracy check
y_pred = model.predict(X_test)

# print accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# save model
joblib.dump(model, "model.pkl")

print("✅ model.pkl created successfully")