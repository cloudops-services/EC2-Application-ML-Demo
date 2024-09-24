import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from src.data import download_data
from src.utils import extract_zipfile

DATASET_PATH = os.path.join("data", "car.data")
NAMES = ["buying", "maint", "doors", "person", "lug_boot", "safety", "class"]

# Download Data

zipfile_path = download_data()
extract_zipfile(zipfile_path=zipfile_path, output="data")

# Split features and target variables

raw_df = pd.read_csv(DATASET_PATH, sep=",", header=None, names=NAMES)
for name in NAMES[:-1]:
    raw_df[name] = raw_df[name].astype("category")

X = raw_df.iloc[:, :-1]
y = raw_df.iloc[:, -1]
print(f"# features = {X.shape[0]}")
print(f"# samples: {X.shape[1]}")

# Encoding the features and saving the encoding files to be used in the application

X_encoded = pd.get_dummies(X, columns=NAMES[:-1]).astype(int)
y_enc = OrdinalEncoder().fit(y.values.reshape(-1, 1))
y_encoded = y_enc.transform(y.values.reshape(-1, 1)).flatten()
with open("y_enc.pkl", "wb") as file:
    pickle.dump(y_enc, file)

dummy_columns = X_encoded.columns
with open("dummy_columns.pkl", "wb") as file:
    pickle.dump(dummy_columns, file)

# Split train and test (80-20)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42
)

print("--- Train ---")
print(f"#samples = {X_train.shape[0]}")
print("--- Test ---")
print(f"#samples = {X_test.shape[0]}")

# Creating model and training

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

with open("trained_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Evaluating the model (acc ~ 0.97 and precision ~ 0.85)

y_pred = model.predict(X_test)
acc = accuracy_score(y_true=y_test, y_pred=y_pred)
prec = precision_score(y_true=y_test, y_pred=y_pred, average="macro")
print(f"accuracy = {acc} | precision = {prec}")
