# train_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# ---------- Example synthetic dataset ----------
rng = np.random.RandomState(42)
n = 500

hours = rng.normal(10, 4, size=n).clip(0)   # study hours
attendance = rng.normal(85, 10, size=n).clip(40,100)  # attendance %
cgpa = rng.normal(7, 1.2, size=n).clip(0,10)  # cgpa
assignments = rng.randint(4, 11, size=n)   # assignments completed
courses = rng.choice(["B.Tech", "MBA", "BCA", "MCA"], size=n)

# Simulated final score
final = (0.5*cgpa*10 + 0.8*hours + 0.2*attendance + 1.5*assignments) / 2.8
final = final + rng.normal(0, 6, size=n)
final = final.clip(0,100)

df = pd.DataFrame({
    "hours": hours,
    "attendance": attendance,
    "cgpa": cgpa,
    "assignments": assignments,
    "course": courses,
    "final_score": final
})

# ---------- Features / Target ----------
X = df[["hours","attendance","cgpa","assignments","course"]]
y = df["final_score"]

# Preprocessing: OneHotEncode 'course'
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), ["course"]),
    ("num", "passthrough", ["hours","attendance","cgpa","assignments"])
])

# Pipeline
model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(f"Test MSE: {mse:.2f}, R2: {r2:.3f}")

# Save model
joblib.dump(model, "model.joblib")
print("✅ Saved model to model.joblib")
