import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =====================================================
# SETTINGS
# =====================================================

INPUT_FILE = r"/Users/mohammad/University/Bachelor Project/Final/Data/Stand/Master Features/MASTER_Features_Stand.xlsx"

OUTPUT_FILE = r"/Users/mohammad/University/Bachelor Project/Final/Data/Stand/Master Features/PCA_Method_Comparison.xlsx"

PCA_VARIANCE = 0.95

TRAIN_PER_CLASS = 20
RANDOM_SEED = 42

# =====================================================
# LOAD DATA
# =====================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_excel(INPUT_FILE)

meta_cols = [
    "label",
    "source_file",
    "prefix",
    "idx"
]

feature_cols = [
    c for c in df.columns
    if c not in meta_cols
]

X = df[feature_cols].copy()
y = df["label"].astype(int).values

for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))

X = X.values

print(f"Dataset shape: {X.shape}")

# =====================================================
# TRAIN SPLIT (same strategy as your classifier code)
# =====================================================

rng = np.random.default_rng(RANDOM_SEED)

train_idx = []

for cls in np.unique(y):

    idx = np.where(y == cls)[0]

    idx = rng.permutation(idx)

    train_idx.extend(
        idx[:TRAIN_PER_CLASS]
    )

train_idx = np.array(train_idx)

X_train = X[train_idx]

print(f"Train-only shape: {X_train.shape}")

# =====================================================
# PCA FUNCTION
# =====================================================

def fit_pca(X_data):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_data)

    pca = PCA(
        n_components=PCA_VARIANCE,
        random_state=42
    )

    pca.fit(X_scaled)

    return pca

# =====================================================
# FEATURE NAME PARSER
# =====================================================

def parse_feature_name(feature_name):

    parts = feature_name.split("_")

    statistic = parts[0]

    axis = parts[-1]

    sensor = "_".join(parts[1:-1])

    return statistic, sensor, axis

# =====================================================
# FEATURE CONTRIBUTIONS
# =====================================================

def feature_contributions(pca, feature_names):

    weights = pca.explained_variance_ratio_

    scores = []

    for i, feat in enumerate(feature_names):

        score = np.sum(
            (pca.components_[:, i] ** 2)
            * weights
        )

        statistic, sensor, axis = parse_feature_name(feat)

        scores.append({
            "Feature": feat,
            "Statistic": statistic,
            "Sensor": sensor,
            "Axis": axis,
            "Contribution": score
        })

    out = pd.DataFrame(scores)

    out["Contribution"] = (
        out["Contribution"]
        /
        out["Contribution"].sum()
        * 100
    )

    return out

# =====================================================
# GROUP CONTRIBUTIONS
# =====================================================

def group_contribution(df_feat, column):

    out = (
        df_feat
        .groupby(column)["Contribution"]
        .sum()
        .reset_index()
    )

    out = out.sort_values(
        "Contribution",
        ascending=False
    )

    return out

# =====================================================
# FIT BOTH PCAs
# =====================================================

print("\nFitting PCA on ALL DATA...")
pca_all = fit_pca(X)

print("Fitting PCA on TRAIN DATA ONLY...")
pca_train = fit_pca(X_train)

print(
    f"\nAll-data PCs retained: {pca_all.n_components_}"
)

print(
    f"Train-only PCs retained: {pca_train.n_components_}"
)

# =====================================================
# VARIANCE COMPARISON
# =====================================================

max_pc = max(
    len(pca_all.explained_variance_ratio_),
    len(pca_train.explained_variance_ratio_)
)

variance_df = pd.DataFrame({
    "PC": [
        f"PC{i+1}"
        for i in range(max_pc)
    ]
})

variance_df["All_Data_%"] = np.nan
variance_df["Train_Only_%"] = np.nan

variance_df.loc[
    :len(pca_all.explained_variance_ratio_) - 1,
    "All_Data_%"
] = pca_all.explained_variance_ratio_ * 100

variance_df.loc[
    :len(pca_train.explained_variance_ratio_) - 1,
    "Train_Only_%"
] = pca_train.explained_variance_ratio_ * 100

# =====================================================
# FEATURE CONTRIBUTIONS
# =====================================================

feat_all = feature_contributions(
    pca_all,
    feature_cols
)

feat_train = feature_contributions(
    pca_train,
    feature_cols
)

# =====================================================
# SENSOR CONTRIBUTION
# =====================================================

sensor_all = group_contribution(
    feat_all,
    "Sensor"
).rename(
    columns={
        "Contribution": "All_Data_%"
    }
)

sensor_train = group_contribution(
    feat_train,
    "Sensor"
).rename(
    columns={
        "Contribution": "Train_Only_%"
    }
)

sensor_df = sensor_all.merge(
    sensor_train,
    on="Sensor",
    how="outer"
).fillna(0)

sensor_df["Difference"] = (
    sensor_df["All_Data_%"]
    -
    sensor_df["Train_Only_%"]
)

sensor_df = sensor_df.sort_values(
    "Difference",
    key=np.abs,
    ascending=False
)

# =====================================================
# AXIS CONTRIBUTION
# =====================================================

axis_all = group_contribution(
    feat_all,
    "Axis"
).rename(
    columns={
        "Contribution": "All_Data_%"
    }
)

axis_train = group_contribution(
    feat_train,
    "Axis"
).rename(
    columns={
        "Contribution": "Train_Only_%"
    }
)

axis_df = axis_all.merge(
    axis_train,
    on="Axis",
    how="outer"
).fillna(0)

axis_df["Difference"] = (
    axis_df["All_Data_%"]
    -
    axis_df["Train_Only_%"]
)

axis_df = axis_df.sort_values(
    "Difference",
    key=np.abs,
    ascending=False
)

# =====================================================
# STATISTIC CONTRIBUTION
# =====================================================

stat_all = group_contribution(
    feat_all,
    "Statistic"
).rename(
    columns={
        "Contribution": "All_Data_%"
    }
)

stat_train = group_contribution(
    feat_train,
    "Statistic"
).rename(
    columns={
        "Contribution": "Train_Only_%"
    }
)

stat_df = stat_all.merge(
    stat_train,
    on="Statistic",
    how="outer"
).fillna(0)

stat_df["Difference"] = (
    stat_df["All_Data_%"]
    -
    stat_df["Train_Only_%"]
)

stat_df = stat_df.sort_values(
    "Difference",
    key=np.abs,
    ascending=False
)

# =====================================================
# FEATURE DIFFERENCES
# =====================================================

feature_df = feat_all[
    ["Feature", "Contribution"]
].rename(
    columns={
        "Contribution": "All_Data_%"
    }
)

feature_df = feature_df.merge(
    feat_train[
        ["Feature", "Contribution"]
    ].rename(
        columns={
            "Contribution": "Train_Only_%"
        }
    ),
    on="Feature"
)

feature_df["Difference"] = (
    feature_df["All_Data_%"]
    -
    feature_df["Train_Only_%"]
)

feature_df["Abs_Difference"] = (
    feature_df["Difference"].abs()
)

feature_df = feature_df.sort_values(
    "Abs_Difference",
    ascending=False
)

# =====================================================
# SAVE
# =====================================================

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    variance_df.to_excel(
        writer,
        sheet_name="Variance_Comparison",
        index=False
    )

    sensor_df.to_excel(
        writer,
        sheet_name="Sensor_Contribution",
        index=False
    )

    axis_df.to_excel(
        writer,
        sheet_name="Axis_Contribution",
        index=False
    )

    stat_df.to_excel(
        writer,
        sheet_name="Statistic_Contribution",
        index=False
    )

    feature_df.to_excel(
        writer,
        sheet_name="Top_Changed_Features",
        index=False
    )

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(f"Saved:\n{OUTPUT_FILE}")