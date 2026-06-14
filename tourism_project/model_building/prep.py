# Data cleaning, encoding, train/test split, and upload to HF
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_REPO = "LeonKennedy007/tourism-package-data"
DATASET_PATH = f"hf://datasets/{DATASET_REPO}/tourism.csv"

# Load dataset from Hugging Face
df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded: {df.shape}")

# Drop identifier columns (not useful for modeling)
df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True)

# Encode categorical features
cat_cols = ["TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation"]
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Separate features and target
TARGET = "ProdTaken"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# Save splits locally
X_train.to_csv("Xtrain.csv", index=False)
X_test.to_csv("Xtest.csv", index=False)
y_train.to_csv("ytrain.csv", index=False)
y_test.to_csv("ytest.csv", index=False)

# Upload splits back to Hugging Face
for fname in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=fname,
        path_in_repo=fname,
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"Uploaded: {fname}")

print("Data preparation complete.")
