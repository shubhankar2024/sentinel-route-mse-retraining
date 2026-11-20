from pathlib import Path
import pandas as pd

from src.config import DATA_RAW_PATH, FEATURE_COLS, TARGET_COL
from src.train import RAW_TARGET_COL, LABEL_MAP


def prepare_and_validate_batch(batch_path: Path) -> pd.DataFrame:
    """Validate an incoming labelled batch CSV (Task A)."""
    path = Path(batch_path)
    if not path.exists():
        raise FileNotFoundError(f"Batch file not found: {path}")

    df = pd.read_csv(path)
    if RAW_TARGET_COL in df.columns and TARGET_COL not in df.columns:
        df = df.rename(columns={RAW_TARGET_COL: TARGET_COL})

    required = FEATURE_COLS + [TARGET_COL]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Batch is missing required columns: {missing_cols}")

    print("\n--- Task A: New Batch Validation ---")
    print(f"Initial rows: {len(df):,} | Columns: {df.shape[1]}")

    # Missing values check before cleanup
    null_counts = df[required].isna().sum()
    non_zero_nulls = null_counts[null_counts > 0].to_dict()
    print(f"Missing values per column (non-zero): {non_zero_nulls if non_zero_nulls else 'None'}")

    # Normalize string labels if needed and drop rows missing features/target
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.lower().map(LABEL_MAP)
    df = df.dropna(subset=required).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Duplicate check (exact row duplicates across required columns)
    dup_count = int(df.duplicated(subset=required).sum())
    print(f"Duplicate rows found: {dup_count}")
    if dup_count > 0:
        df = df.drop_duplicates(subset=required).reset_index(drop=True)

    # Dataset size & target distribution
    total = len(df)
    counts = df[TARGET_COL].value_counts().to_dict()
    c0 = int(counts.get(0, 0))
    c1 = int(counts.get(1, 0))
    pct0 = (c0 / total * 100.0) if total else 0.0
    pct1 = (c1 / total * 100.0) if total else 0.0

    print(f"Validated batch size: {total:,} rows")
    print(f"Target distribution ({TARGET_COL}): 0 (normal)={c0} ({pct0:.2f}%), 1 (anomaly)={c1} ({pct1:.2f}%)")
    return df


def create_sample_batch(out_path: Path = Path("data/batches/new_batch.csv"), n_rows: int = 300) -> Path:
    """Helper to generate a sample labelled batch from the dataset rule if needed."""
    src_df = pd.read_csv(DATA_RAW_PATH)
    if RAW_TARGET_COL in src_df.columns and TARGET_COL not in src_df.columns:
        src_df = src_df.rename(columns={RAW_TARGET_COL: TARGET_COL})
    sample = src_df.sample(n=min(n_rows, len(src_df)), random_state=101).copy()
    # Add 2 intentional duplicate rows so duplicate detection is visibly demonstrated
    if len(sample) >= 2:
        sample = pd.concat([sample, sample.iloc[:2]], ignore_index=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    return out
