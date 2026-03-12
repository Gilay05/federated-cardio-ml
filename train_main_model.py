import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
train = pd.read_csv("venv/central_server/set1_train.csv")
test = pd.read_csv("venv/central_server/set1_test.csv")

X_train = train.drop(columns=["cardio"])
y_train = train["cardio"]

X_test = test.drop(columns=["cardio"])
y_test = test["cardio"]

# Train model
model = GradientBoostingClassifier(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=3
)

model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print("Main Model Accuracy:", acc)

# Save model
joblib.dump(model, "venv/models/main_model.pkl")

print("Main model saved.")