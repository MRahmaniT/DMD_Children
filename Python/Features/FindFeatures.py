import os
import glob
import numpy as np
import pandas as pd

# =====================================================
# SETTINGS
# =====================================================

# 1. Choose task : Stand / Sit_To_Stand / Jump / Rise_Head / Climb_Box_Right_foot
TASK = "Rise_Head" 

# 2. Did you use action detector on your data or not
DETECTED = True

# 3. Choose range of indexes for above task (Example for range 2 to 26))
INDEX_RANGE_MIN = 2
INDEX_RANGE_MAX = 27


# 4. Coose Labels
LABELS = (0, 1, 2)

# 5. If your excels have no headers or wrong headers, enable this:
FORCE_HEADERS = False

# 6. Enable to save all of features in one master file
SAVE_MASTER = True

# 7. Path
if DETECTED :
    INPUT_DIR = r"/Users/mohammad/University/Bachelor Project/Final/DetectedActionData/" + TASK
    OUTPUT_DIR = r"/Users/mohammad/University/Bachelor Project/Final/DetectedActionData/" + TASK + "/Features" 
    MASTER_DIR = r"/Users/mohammad/University/Bachelor Project/Final/DetectedActionData/" + TASK + "/Master Features" 
else:
    INPUT_DIR = r"/Users/mohammad/University/Bachelor Project/Final/Data/" + TASK
    OUTPUT_DIR = r"/Users/mohammad/University/Bachelor Project/Final/Data/" + TASK + "/Features" 
    MASTER_DIR = r"/Users/mohammad/University/Bachelor Project/Final/Data/" + TASK + "/Master Features" 
        
# -----------------------------
# 1) Your headers (ground truth)
# -----------------------------
VARNAMES = [
    'Head Time', 'Head Ax','Head Ay','Head Az','Head Gx','Head Gy','Head Gz',
    'Right Hand Time', 'Right Hand Ax','Right Hand Ay','Right Hand Az','Right Hand Gx','Right Hand Gy','Right Hand Gz',
    'Left Hand Time', 'Left Hand Ax','Left Hand Ay','Left Hand Az','Left Hand Gx','Left Hand Gy','Left Hand Gz',
    'Right Foot Time', 'Right Foot Ax','Right Foot Ay','Right Foot Az','Right Foot Gx','Right Foot Gy','Right Foot Gz',
    'Left Foot Time', 'Left Foot Ax','Left Foot Ay','Left Foot Az','Left Foot Gx','Left Foot Gy','Left Foot Gz'
]

TIME_COLS = [c for c in VARNAMES if "Time" in c]  # remove all time columns

def clean_name(col: str) -> str:
    # "Right Hand Ax" -> "right_hand_ax"
    s = col.strip().lower()
    s = s.replace("-", " ")
    s = "_".join(s.split())
    return s

# -----------------------------
# 2) Feature helpers
# -----------------------------
def _safe_arr(x: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr

def _entropy_hist(arr: np.ndarray, bins: int = 20) -> float:
    if arr.size == 0:
        return np.nan
    hist, _ = np.histogram(arr, bins=bins, density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return np.nan
    return float(-np.sum(hist * np.log2(hist)))

def _zero_crossings(arr: np.ndarray) -> float:
    if arr.size < 2:
        return np.nan
    s = np.sign(arr)
    return float(np.sum(s[1:] * s[:-1] < 0))

def _autocorr_lag1(arr: np.ndarray) -> float:
    if arr.size < 2:
        return np.nan
    x = arr - np.mean(arr)
    denom = np.dot(x, x)
    if denom == 0:
        return 0.0
    return float(np.dot(x[:-1], x[1:]) / denom)

# -----------------------------
# 3) Extract features, naming like: mean_head_ax
# -----------------------------
def extract_features(df: pd.DataFrame) -> dict:
    out = {}

    # Ensure columns names match your varNames if possible
    # If file already has correct headers, this won't change anything.
    # If file has unnamed columns, you can force them:
    # df.columns = VARNAMES[:len(df.columns)]

    # Drop time columns by name (also robust to case)
    cols_lower = {c.lower(): c for c in df.columns}
    for t in TIME_COLS:
        if t.lower() in cols_lower:
            df = df.drop(columns=[cols_lower[t.lower()]])

    # Convert numeric candidates (sometimes Excel reads as object)
    work = df.copy()
    for c in work.columns:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    # Keep only non-time numeric columns
    numeric_cols = [c for c in work.columns if c not in TIME_COLS and work[c].notna().any()]

    for col in numeric_cols:
        arr = _safe_arr(work[col])
        base = clean_name(col)

        # if empty after cleaning
        if arr.size == 0:
            stats = {
                "count": 0,
                "mean": np.nan, "std": np.nan, "var": np.nan,
                "min": np.nan, "max": np.nan, "range": np.nan,
                "median": np.nan, "q25": np.nan, "q75": np.nan, "iqr": np.nan,
                "mad": np.nan, "mean_abs": np.nan, "rms": np.nan, "energy": np.nan,
                "cv": np.nan, "skew": np.nan, "kurtosis": np.nan,
                "entropy": np.nan, "zero_crossings": np.nan,
                "diff_mean": np.nan, "diff_std": np.nan,
                "autocorr_lag1": np.nan
            }
        else:
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            var = float(np.var(arr, ddof=1)) if arr.size > 1 else 0.0
            mn = float(np.min(arr)); mx = float(np.max(arr))
            med = float(np.median(arr))
            q25 = float(np.percentile(arr, 25))
            q75 = float(np.percentile(arr, 75))
            iqr = float(q75 - q25)
            mad = float(np.median(np.abs(arr - med)))
            mean_abs = float(np.mean(np.abs(arr)))
            rms = float(np.sqrt(np.mean(arr ** 2)))
            energy = float(np.sum(arr ** 2))
            cv = float(std / mean) if mean != 0 else np.nan

            if arr.size > 2 and std > 0:
                z = (arr - mean) / std
                skew = float(np.mean(z ** 3))
            else:
                skew = np.nan

            if arr.size > 3 and std > 0:
                z = (arr - mean) / std
                kurt = float(np.mean(z ** 4) - 3.0)  # excess kurtosis
            else:
                kurt = np.nan

            dif = np.diff(arr) if arr.size > 1 else np.array([])
            diff_mean = float(np.mean(dif)) if dif.size else np.nan
            diff_std = float(np.std(dif, ddof=1)) if dif.size > 1 else (0.0 if dif.size == 1 else np.nan)

            stats = {
                "count": int(arr.size),
                "mean": mean, "std": std, "var": var,
                "min": mn, "max": mx, "range": float(mx - mn),
                "median": med, "q25": q25, "q75": q75, "iqr": iqr,
                "mad": mad, "mean_abs": mean_abs, "rms": rms, "energy": energy,
                "cv": cv, "skew": skew, "kurtosis": kurt,
                "entropy": _entropy_hist(arr, bins=20),
                "zero_crossings": _zero_crossings(arr),
                "diff_mean": diff_mean, "diff_std": diff_std,
                "autocorr_lag1": _autocorr_lag1(arr)
            }

        # Name like: mean_head_ax, std_head_ax, ...
        for feat, val in stats.items():
            out[f"{feat}_{base}"] = val

    return out

# -----------------------------
# 4) File finder (robust)
# -----------------------------
def find_file(input_dir: str, prefix: str, label: int, idx: int) -> str | None:
    if DETECTED:
        KIND = "DetectedAction_Filtered"
    else:
        KIND = "Filtered"
    patterns = [
        os.path.join(input_dir, f"KIND.{prefix}_{label}.{idx:02d}.xlsx"),
        os.path.join(input_dir, f"KIND.{prefix}_{label}.{idx}.xlsx"),
        os.path.join(input_dir, f"*{prefix}*_{label}.{idx:02d}*.xlsx"),
        os.path.join(input_dir, f"*{prefix}*_{label}.{idx}*.xlsx"),
    ]
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            hits = sorted(hits, key=lambda x: (len(os.path.basename(x)), x))
            return hits[0]
    return None

# -----------------------------
# 5) Main pipeline: 75 files -> 75 one-row excels (+ optional master)
# -----------------------------
def process_dataset(
    input_dir: str,
    output_dir: str,
    master_dir: str,
    prefix: str,
    idx_range = range(INDEX_RANGE_MIN, INDEX_RANGE_MAX),
    labels = LABELS,
    force_headers: bool =FORCE_HEADERS,
    save_master: bool = SAVE_MASTER
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(master_dir, exist_ok=True)
    master_rows = []
    missing = []

    for idx in idx_range:
        for lab in labels:
            fpath = find_file(input_dir, prefix, lab, idx)
            if fpath is None:
                missing.append((prefix, lab, idx))
                continue

            df = pd.read_excel(fpath)

            if force_headers:
                df.columns = VARNAMES[:len(df.columns)]

            feats = extract_features(df)
            feats["label"] = int(lab)  # last column desired

            out_df = pd.DataFrame([feats])

            base = os.path.splitext(os.path.basename(fpath))[0]
            out_path = os.path.join(output_dir, f"Features_{base}.xlsx")
            out_df.to_excel(out_path, index=False)

            # master
            row = feats.copy()
            row["source_file"] = os.path.basename(fpath)
            row["idx"] = int(idx)
            row["prefix"] = prefix
            master_rows.append(row)

    if save_master and master_rows:
        master_df = pd.DataFrame(master_rows)

        meta_cols = [c for c in ["source_file", "prefix", "idx"] if c in master_df.columns]
        feature_cols = [c for c in master_df.columns if c not in meta_cols + ["label"]]
        master_df = master_df[meta_cols + feature_cols + ["label"]]

        master_path = os.path.join(master_dir, f"MASTER_Features_{prefix}.xlsx")
        master_df.to_excel(master_path, index=False)

    return {"processed": len(master_rows), "missing": missing, "output_dir": output_dir}


if __name__ == "__main__":    

    result = process_dataset(
        input_dir = INPUT_DIR,
        output_dir = OUTPUT_DIR,
        master_dir = MASTER_DIR,
        prefix = TASK,
        idx_range = range(INDEX_RANGE_MIN, INDEX_RANGE_MAX),
        labels = LABELS,
        force_headers = FORCE_HEADERS,
        save_master = SAVE_MASTER
    )

    print("Processed:", result["processed"])
    if result["missing"]:
        print("Missing files:", result["missing"])
    print("Saved in:", result["output_dir"])
