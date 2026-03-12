import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load main model
model = joblib.load("venv/models/main_model.pkl")

train = pd.read_csv("hospital1/set2_train.csv")
test = pd.read_csv("hospital1/set2_test.csv")

X_test = test.drop(columns=["cardio"])
y_test = test["cardio"]

baseline = 0.731

# ----------------------
# Step 1: Test 1500
# ----------------------
X_initial = X_test.iloc[:1500]
y_initial = y_test.iloc[:1500]

pred_initial = model.predict(X_initial)

initial_acc = accuracy_score(y_initial, pred_initial)

print("Hospital 1 Accuracy (1500 samples):", initial_acc)

# ----------------------
# Scenario A: Accuracy reduced
# ----------------------
if initial_acc < baseline:

    print("Accuracy reduced → retraining model")

    X_train = train.drop(columns=["cardio"])
    y_train = train["cardio"]

    model.fit(X_train, y_train)

# ----------------------
# Scenario B: Accuracy NOT reduced
# ----------------------
else:

    print("Accuracy not reduced → adding 200 more test samples")

    X_extended = X_test.iloc[:1700]
    y_extended = y_test.iloc[:1700]

    pred_extended = model.predict(X_extended)

    extended_acc = accuracy_score(y_extended, pred_extended)

    print("Hospital 1 Accuracy (1700 samples):", extended_acc)

    print("Retraining model as per protocol")

    X_train = train.drop(columns=["cardio"])
    y_train = train["cardio"]

    model.fit(X_train, y_train)

# ----------------------
# Final evaluation
# ----------------------
pred_final = model.predict(X_test)

final_acc = accuracy_score(y_test, pred_final)

print("Hospital 1 Final Accuracy (3000 samples):", final_acc)

# Save model
joblib.dump(model, "venv/models/hospital1_v2.pkl")

print("hospital1_v2 model saved.")