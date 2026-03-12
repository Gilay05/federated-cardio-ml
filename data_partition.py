import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

# load dataset
df = pd.read_csv("cardiovascular_disease_dataset.csv")
target_col = "cardio"

# sanity check
assert df.shape[0] == 70000, "Dataset must have 70,000 rows"

# desired totals
targets = {
    "set1": {"total": 34400, "train": 27400, "test": 7000},
    "set2": {"total": 15500, "train": 12500, "test": 3000},
    "set3": {"total": 20100, "train": 16100, "test": 4000},
}
# helper to get counts per class for a given total using stratified proportions
def class_counts_for_total(df, total):
    # proportion of each class in full df
    counts = df[target_col].value_counts(normalize=False).sort_index()
    class_props = (counts / counts.sum()).to_dict()
    # compute integer counts per class (rounding using floor then distribute remainder)
    raw = {cls: int(np.floor(prop * total)) for cls, prop in class_props.items()}
    rem = total - sum(raw.values())
    # distribute remainder by largest fractional parts (tie-breaker by class label)
    frac = {cls: (prop * total) - raw[cls] for cls, prop in class_props.items()}
    for cls in sorted(frac, key=lambda k: -frac[k])[:rem]:
        raw[cls] += 1
    return raw  # dict: {0: n0, 1: n1}

# Function to sample a stratified subset with exact per-class counts
def stratified_sample_counts(df, counts_per_class, random_state=42):
    parts = []
    for cls, count in counts_per_class.items():
        cls_df = df[df[target_col] == cls]
        samples = cls_df.sample(n=count, random_state=random_state)
        parts.append(samples)
    return pd.concat(parts).sample(frac=1.0, random_state=random_state)

remaining = df.copy()

# Build set1
set1_counts = class_counts_for_total(df, targets['set1']['total'])
set1 = stratified_sample_counts(remaining, set1_counts, random_state=1)
remaining = remaining.drop(set1.index)

# Build set2 (from remaining)
set2_counts = class_counts_for_total(remaining, targets['set2']['total'])
set2 = stratified_sample_counts(remaining, set2_counts, random_state=2)
remaining = remaining.drop(set2.index)

# Build set3 (rest)
set3 = remaining.copy()
assert set3.shape[0] == targets['set3']['total']

# Now split each set into train/test using exact sizes
def split_exact(set_df, train_count, test_count, random_state=42):
    assert set_df.shape[0] == train_count + test_count
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_count, random_state=random_state)
    X = set_df.drop(columns=[target_col])
    y = set_df[target_col]
    train_idx, test_idx = next(sss.split(X, y))
    return set_df.iloc[train_idx].reset_index(drop=True), set_df.iloc[test_idx].reset_index(drop=True)

set1_train, set1_test = split_exact(set1, targets['set1']['train'], targets['set1']['test'], random_state=11)
set2_train, set2_test = split_exact(set2, targets['set2']['train'], targets['set2']['test'], random_state=12)
set3_train, set3_test = split_exact(set3, targets['set3']['train'], targets['set3']['test'], random_state=13)

# Save CSVs for reproducibility
set1_train.to_csv("set1_train.csv", index=False)
set1_test.to_csv("set1_test.csv", index=False)
set2_train.to_csv("set2_train.csv", index=False)
set2_test.to_csv("set2_test.csv", index=False)
set3_train.to_csv("set3_train.csv", index=False)
set3_test.to_csv("set3_test.csv", index=False)

print("Partitioning done.")
print("Set1:", set1_train.shape[0], set1_test.shape[0])
print("Set2:", set2_train.shape[0], set2_test.shape[0])
print("Set3:", set3_train.shape[0], set3_test.shape[0])