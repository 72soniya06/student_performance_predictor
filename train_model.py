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

# Features
attendance = rng.normal(85, 10, size=n).clip(40, 100)   # attendance %
cgpa = rng.normal(7, 1.2, size=n).clip(0, 10)           # cgpa
assignments = rng.randint(4, 11, size=n)                # assignments completed
internal_marks = rng.randint(10, 30, size=n)            # internal marks (0–30)
courses = rng.choice(["B.Tech", "MBA", "BCA", "MCA"], size=n)

# Extra info columns (not used in training)
names = [f"Student_{i}" for i in range(n)]
roll_nos = [f"R{i:03d}" for i in range(n)]
branches = rng.choice(["CSE", "ECE", "IT", "Civil", "Mechanical"], size=n)
sections = rng.choice(["A", "B", "C", "D"], size=n)

# Simulated final score (internal marks + cgpa + attendance + assignments)
final = (0.5 * cgpa * 10 + 0.3 * attendance + 1.2 * assignments + 2.0 * internal_marks) / 2.5
final = final + rng.normal(0, 5, size=n)   # add noise
final = final.clip(0, 100)

# DataFrame
df = pd.DataFrame({
    "name": names,
    "roll_no": roll_nos,
    "attendance": attendance,
    "cgpa": cgpa,
    "assignments": assignments,
    "internal_marks": internal_marks,
    "course": courses,
    "branch": branches,
    "section": sections,
    "final_score": final
})

# ---------- Features / Target ----------
X = df[["attendance", "cgpa", "assignments", "internal_marks", "course"]]
y = df["final_score"]

# Preprocessing: OneHotEncode 'course'
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), ["course"]),
    ("num", "passthrough", ["attendance", "cgpa", "assignments", "internal_marks"])
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
