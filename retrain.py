import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib

# 1. Generate dummy training data
np.random.seed(42)
n = 100
X = pd.DataFrame({
    "Weight": np.random.randint(1000, 10000, size=n),
    "Arm": np.random.randint(10, 60, size=n),
    "WindSpeed": np.random.uniform(0, 50, size=n),
    "Altitude": np.random.randint(0, 20000, size=n),
})

# 2. Define target variable (Turbulence Class) based on WindSpeed
turbulence_score = X["WindSpeed"] / 45.0
y = pd.cut(
    turbulence_score,
    bins=[0, 0.3, 0.7, 1.0],
    labels=["Low", "Medium", "High"],
    include_lowest=True,
    right=True
)

# 3. Drop rows with NaN labels (just in case)
valid = ~y.isna()
X = X[valid]
y = y[valid]

# 4. Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# 5. Save model
joblib.dump(model, "model_turbulence.pkl")
print("✅ model_turbulence.pkl has been saved successfully.")
