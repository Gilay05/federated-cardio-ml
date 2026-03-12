import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Load hospital models
h1 = joblib.load("venv/models/hospital1_v2.pkl")
h2 = joblib.load("venv/models/hospital2_v2.pkl")

# Create federated model
federated_model = VotingClassifier(
    estimators=[
        ("hospital1", h1),
        ("hospital2", h2)
    ],
    voting="soft"
)

# Train aggregator using Set1 training data
train = pd.read_csv("central_server/set1_train.csv")

X_train = train.drop(columns=["cardio"])
y_train = train["cardio"]

federated_model.fit(X_train, y_train)

# Evaluate
test = pd.read_csv("central_server/set1_test.csv")

X_test = test.drop(columns=["cardio"])
y_test = test["cardio"]

pred = federated_model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("Federated Model Accuracy:", acc)

# Save new global model
joblib.dump(federated_model, "venv/models/main_model_v2.pkl")

print("main_model_v2 saved.")