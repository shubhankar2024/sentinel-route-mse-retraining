from datetime import datetime, timezone
import json
from pathlib import Path
from mlflow.tracking import MlflowClient

from src.config import MIN_PROMOTION_F1, MODEL_ALIAS


def evaluate_quality_gate(
    champion_metrics: dict,
    candidate_metrics: dict,
    min_f1: float = MIN_PROMOTION_F1,
    output_path: Path = Path("artifacts/quality_gate_decision.json"),
) -> dict:
    """Compare existing champion vs candidate and record promotion eligibility (Task D)."""
    champ_f1 = float(champion_metrics.get("f1_score", 0.0))
    cand_f1 = float(candidate_metrics.get("f1_score", 0.0))

    meets_floor = cand_f1 >= min_f1
    beats_or_ties_champion = cand_f1 >= champ_f1
    eligible = meets_floor and beats_or_ties_champion

    decision = "ELIGIBLE_FOR_PROMOTION" if eligible else "RETAINED_CHAMPION"
    reasons = []
    if not meets_floor:
        reasons.append(f"candidate F1 ({cand_f1:.4f}) < minimum floor ({min_f1:.4f})")
    if not beats_or_ties_champion:
        reasons.append(f"candidate F1 ({cand_f1:.4f}) < champion F1 ({champ_f1:.4f})")
    if not reasons:
        reasons.append(
            f"candidate F1 ({cand_f1:.4f}) >= floor ({min_f1:.4f}) and >= champion F1 ({champ_f1:.4f})"
        )

    record = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "primary_metric": "f1_score",
        "min_promotion_f1": min_f1,
        "champion_metrics": champion_metrics,
        "candidate_metrics": candidate_metrics,
        "eligible_for_promotion": eligible,
        "decision": decision,
        "reason": "; ".join(reasons),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print("\n--- Task D: Quality Gate Decision ---")
    print(f"Champion F1 : {champ_f1:.4f} | Candidate F1: {cand_f1:.4f} | Floor: {min_f1:.4f}")
    print(f"Decision    : {decision} ({record['reason']})")
    print(f"Saved gate log to {out}")
    return record


def promote_candidate_version(client: MlflowClient, model_name: str, version: str) -> None:
    """Explicit promotion helper (only call when eligible and promotion is requested)."""
    client.set_registered_model_alias(model_name, MODEL_ALIAS, str(version))
    print(f"[{model_name}] Promoted version {version} to @{MODEL_ALIAS}")
