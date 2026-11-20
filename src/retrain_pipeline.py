import argparse
import json
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_RAW_PATH,
    EXPERIMENT_NAME,
    FEATURE_COLS,
    HEAVY_MODEL_NAME,
    MIN_PROMOTION_F1,
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    TARGET_COL,
)
from src.data_prep import create_sample_batch, prepare_and_validate_batch
from src.quality_gate import evaluate_quality_gate, promote_candidate_version
from src.train import build_models, evaluate_model, file_md5, get_git_sha, load_dataset


def score_predictions(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def run_pipeline(batch_path: Path, promote: bool = False) -> None:
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Task A: Validate new incoming batch
    new_df = prepare_and_validate_batch(batch_path)
    X_new = new_df[FEATURE_COLS]
    y_new = new_df[TARGET_COL].astype(int)

    # Task B: Load existing champion model and evaluate on new labelled batch
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    print("\n--- Task B: Evaluate Existing Champion on New Batch ---")
    champion_model = None
    champion_version = None
    try:
        mv = client.get_model_version_by_alias(HEAVY_MODEL_NAME, MODEL_ALIAS)
        champion_version = mv.version
        champion_model = mlflow.sklearn.load_model(f"models:/{HEAVY_MODEL_NAME}@{MODEL_ALIAS}")
        champ_preds = champion_model.predict(X_new)
        champion_batch_metrics = score_predictions(y_new, champ_preds)
        print(f"Loaded {HEAVY_MODEL_NAME}@{MODEL_ALIAS} (v{champion_version}) -> {champion_batch_metrics}")
    except Exception as exc:
        print(f"Warning: Could not load @{MODEL_ALIAS} ({exc}). Using zero baseline metrics.")
        champion_batch_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    # Task C: Combine existing training data + new batch and train candidate model
    print("\n--- Task C: Combine Data, Retrain Candidate & Save Artifact ---")
    base_df = load_dataset(DATA_RAW_PATH)
    combined_df = pd.concat([base_df[FEATURE_COLS + [TARGET_COL]], new_df[FEATURE_COLS + [TARGET_COL]]], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    print(f"Combined dataset size: {len(combined_df):,} rows (base={len(base_df):,}, new={len(new_df):,})")

    X_comb = combined_df[FEATURE_COLS]
    y_comb = combined_df[TARGET_COL].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X_comb, y_comb, test_size=0.20, stratify=y_comb, random_state=42
    )

    candidate_model = build_models()["heavy_pro"]
    candidate_model.fit(X_train, y_train)
    candidate_holdout_metrics = evaluate_model(candidate_model, X_test, y_test)
    candidate_batch_metrics = score_predictions(y_new, candidate_model.predict(X_new))

    # Save candidate model separately
    candidate_artifact_path = artifacts_dir / "candidate_heavy_pro.joblib"
    joblib.dump(candidate_model, candidate_artifact_path)
    print(f"Saved candidate model artifact to {candidate_artifact_path}")

    tags = {
        "git_sha": get_git_sha(),
        "batch_md5": file_md5(Path(batch_path)),
        "workflow": "batch_retrain",
    }

    with mlflow.start_run(run_name=f"candidate-retrain-{tags['git_sha']}"):
        mlflow.set_tags(tags)
        mlflow.log_params({
            "n_combined": len(combined_df),
            "n_new_batch": len(new_df),
            "champion_version": champion_version or "none",
            "min_promotion_f1": MIN_PROMOTION_F1,
        })
        mlflow.log_metrics({
            "champion_new_batch_f1": champion_batch_metrics["f1_score"],
            "candidate_new_batch_f1": candidate_batch_metrics["f1_score"],
            "candidate_holdout_f1": candidate_holdout_metrics["f1_score"],
        })
        mlflow.sklearn.log_model(
            sk_model=candidate_model,
            artifact_path="candidate_model",
            registered_model_name=HEAVY_MODEL_NAME,
            input_example=X_train.head(3),
        )

    versions = client.search_model_versions(f"name='{HEAVY_MODEL_NAME}'")
    latest_ver = str(max(int(v.version) for v in versions))

    # Task D: Quality gate decision (compare on new batch; also report holdout)
    gate_record = evaluate_quality_gate(
        champion_metrics=champion_batch_metrics,
        candidate_metrics=candidate_batch_metrics,
        min_f1=MIN_PROMOTION_F1,
        output_path=artifacts_dir / "quality_gate_decision.json",
    )

    if promote and gate_record["eligible_for_promotion"]:
        promote_candidate_version(client, HEAVY_MODEL_NAME, latest_ver)
    else:
        print(f"Champion alias @{MODEL_ALIAS} left unchanged (promote={promote}, eligible={gate_record['eligible_for_promotion']}).")

    # Task B report saved in artifacts/
    eval_report = {
        "batch_file": str(batch_path),
        "batch_rows": len(new_df),
        "champion_version": champion_version,
        "candidate_version": latest_ver,
        "existing_champion_on_new_batch": champion_batch_metrics,
        "candidate_on_new_batch": candidate_batch_metrics,
        "candidate_on_combined_holdout": candidate_holdout_metrics,
        "quality_gate": gate_record,
    }
    report_path = artifacts_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(eval_report, indent=2), encoding="utf-8")
    print(f"Saved evaluation report to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="New-batch evaluation, retraining, and quality gate pipeline")
    parser.add_argument("--batch", type=Path, default=Path("data/batches/new_batch.csv"))
    parser.add_argument("--generate-sample", action="store_true", help="Create a sample new_batch.csv if missing")
    parser.add_argument("--promote", action="store_true", help="Promote candidate to @champion if eligible")
    args = parser.parse_args()

    if args.generate_sample and not args.batch.exists():
        create_sample_batch(args.batch)

    run_pipeline(batch_path=args.batch, promote=args.promote)


if __name__ == "__main__":
    main()
