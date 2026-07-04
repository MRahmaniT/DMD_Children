import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =====================================================
# SETTINGS
# =====================================================
# Stand / Sit_To_Stand / ...
TASK = "Sit_To_Stand"

INPUT_FILE = r"/Users/mohammad/University/Bachelor Project/Final/Data/" + TASK + "/Master Features/MASTER_Features_" + TASK + ".xlsx"

OUTPUT_FILE = r"/Users/mohammad/University/Bachelor Project/Final/Data/" + TASK + "/Master Features/MASTER_Features_" + TASK + "_PCA95.xlsx"

PCA_VARIANCE = 0.95

# =====================================================
# LOAD DATA
# =====================================================

print("="*70)
print("LOADING DATA")
print("="*70)

df = pd.read_excel(INPUT_FILE)

# -----------------------------------------------------
# keep label
# -----------------------------------------------------

if "label" not in df.columns:
    raise ValueError("label column not found")

y = df["label"].copy()

# -----------------------------------------------------
# remove metadata
# -----------------------------------------------------

meta_cols = [
    "label",
    "source_file",
    "prefix",
    "idx"
]

X = df.drop(
    columns=[c for c in meta_cols if c in df.columns],
    errors="ignore"
).copy()

# -----------------------------------------------------
# numeric conversion
# -----------------------------------------------------

for c in X.columns:
    X[c] = pd.to_numeric(
        X[c],
        errors="coerce"
    )

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median(numeric_only=True)
)

print(f"Original shape: {X.shape}")

# =====================================================
# STANDARDIZE
# =====================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =====================================================
# PCA
# =====================================================

print("\nApplying PCA...")

pca = PCA(
    n_components=PCA_VARIANCE,
    random_state=42
)

X_pca = pca.fit_transform(
    X_scaled
)

print(
    f"Reduced shape: {X_pca.shape}"
)

print(
    f"Explained variance: "
    f"{pca.explained_variance_ratio_.sum()*100:.2f}%"
)

# =====================================================
# CREATE NEW DATAFRAME
# =====================================================

pc_names = [
    f"PC{i+1}"
    for i in range(X_pca.shape[1])
]

df_pca = pd.DataFrame(
    X_pca,
    columns=pc_names
)

df_pca["label"] = y.values

# =====================================================
# SAVE
# =====================================================

df_pca.to_excel(
    OUTPUT_FILE,
    index=False
)

print("\n" + "="*70)
print("DONE")
print("="*70)

print(
    f"Saved:\n{OUTPUT_FILE}"
)

print(
    f"\nOriginal features: {X.shape[1]}"
)

print(
    f"PCA features: {X_pca.shape[1]}"
)

print(
    f"Variance retained: "
    f"{pca.explained_variance_ratio_.sum()*100:.2f}%"
)