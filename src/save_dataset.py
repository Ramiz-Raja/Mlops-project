import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load dataset from sklearn
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target  # 0 = malignant, 1 = benign

# Save as CSV
df.to_csv("breast_cancer_data.csv", index=False)
print("✅ Dataset saved as breast_cancer_data.csv with shape:", df.shape)
