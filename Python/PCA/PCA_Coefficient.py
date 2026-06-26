import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =====================================================
# SETTINGS
# =====================================================

INPUT_FILE = r"/Users/mohammad/University/Bachelor Project/Final/Data/Stand/Master Features/MASTER_Features_Stand.xlsx"

OUTPUT_FILE = r"/Users/mohammad/University/Bachelor Project/Final/Data/Stand/Master Features/PCA_Interpretation_95.xlsx"

PCA_VARIANCE = 0.95

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_excel(INPUT_FILE)

feature_cols = [
    c for c in df.columns
    if c not in ["label", "source_file", "prefix", "idx"]
]

X = df[feature_cols].copy()

for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))

# =====================================================
# STANDARDIZE
# =====================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================
# PCA
# =====================================================

pca = PCA(
    n_components=PCA_VARIANCE,
    random_state=42
)

X_pca = pca.fit_transform(X_scaled)

# =====================================================
# LOADINGS
# =====================================================

loadings = pd.DataFrame(
    pca.components_.T,
    index=feature_cols,
    columns=[
        f"PC{i+1}"
        for i in range(pca.n_components_)
    ]
)

# =====================================================
# VARIANCE TABLE
# =====================================================

variance_df = pd.DataFrame({
    "PC": [
        f"PC{i+1}"
        for i in range(pca.n_components_)
    ],
    "Explained_Variance_%":
        pca.explained_variance_ratio_ * 100,
    "Cumulative_Variance_%":
        np.cumsum(
            pca.explained_variance_ratio_
        ) * 100
})

# =====================================================
# PCA SCORES (OPTIONAL)
# =====================================================

scores_df = pd.DataFrame(
    X_pca,
    columns=[
        f"PC{i+1}"
        for i in range(pca.n_components_)
    ]
)

if "label" in df.columns:
    scores_df["label"] = df["label"].values

# =====================================================
# SAVE
# =====================================================

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    loadings.to_excel(
        writer,
        sheet_name="PCA_Loadings"
    )

    variance_df.to_excel(
        writer,
        sheet_name="Variance",
        index=False
    )

    scores_df.to_excel(
        writer,
        sheet_name="PC_Scores",
        index=False
    )

print("="*70)
print("PCA INTERPRETATION COMPLETE")
print("="*70)
print(f"Original features : {X.shape[1]}")
print(f"Retained PCs       : {pca.n_components_}")
print(
    f"Variance retained  : "
    f"{pca.explained_variance_ratio_.sum()*100:.2f}%"
)
print(f"Saved: {OUTPUT_FILE}")