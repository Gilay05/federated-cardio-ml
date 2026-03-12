import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

model = joblib.load("venv/models/main_model.pkl")

train = pd.read_csv("venv/data/set3_train.csv")
test = pd.read_csv("venv/data/set3_test.csv")

X_test = test.drop(columns=["cardio"])
y_test = test["cardio"]

baseline = 0.731

# ----------------------
# Step 1: Test 3000
# ----------------------
X_initial = X_test.iloc[:3000]
y_initial = y_test.iloc[:3000]

pred_initial = model.predict(X_initial)

initial_acc = accuracy_score(y_initial, pred_initial)

print("Hospital 2 Accuracy (3000 samples):", initial_acc)

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

    X_extended = X_test.iloc[:3200]
    y_extended = y_test.iloc[:3200]

    pred_extended = model.predict(X_extended)

    extended_acc = accuracy_score(y_extended, pred_extended)

    print("Hospital 2 Accuracy (3200 samples):", extended_acc)

    print("Retraining model as per protocol")

    X_train = train.drop(columns=["cardio"])
    y_train = train["cardio"]

    model.fit(X_train, y_train)

# ----------------------
# Final evaluation
# ----------------------
pred_final = model.predict(X_test)

final_acc = accuracy_score(y_test, pred_final)

print("Hospital 2 Final Accuracy (4000 samples):", final_acc)

joblib.dump(model, "venv/models/hospital2_v2.pkl")

print("hospital2_v2 model saved.")