from unittest.mock import MagicMock  # Replace MLflow models with deterministic fakes.

import numpy as np  # Build probability arrays and stable distributions.
import pandas as pd  # Build live-traffic frames for drift tests.
import pytest  # Fixtures and assertions.
from fastapi.testclient import TestClient  # Exercise both apps without a server.

import src.api as api_main  # Imported as a module so fixtures can patch globals.
from src.config import DATA_RAW_PATH, FEATURE_COLS  # Dataset path and model inputs.
from src.ops import evaluate_drift, target_app  # Drift logic and the simulated service.
from src.train import load_dataset  # Dataset loader lives with training.

SAMPLE_PAYLOAD = {  # Valid healthy-looking telemetry shared by API tests.
    "cpu_utilization": 30.0, "memory_usage": 40.0, "disk_io": 12.0, "network_latency": 20.0,
    "process_count": 200, "thread_count": 700, "context_switches": 500,
    "cache_miss_rate": 0.04, "temperature": 45.0, "power_consumption": 120.0, "uptime": 50.0,
}


# --- Data -------------------------------------------------------------------

def test_kaggle_dataset_loads_and_normalises_target():  # Real CSV must match the contract.
    df = load_dataset()  # Load the tracked dataset.
    assert DATA_RAW_PATH.name == "Big_data_dataset.csv"  # Expected file name.
    assert set(df["system_status"].unique()) <= {0, 1}  # Binary labels only.
    assert len(df) == 10_000  # No rows silently dropped.
    assert all(col in df.columns for col in FEATURE_COLS)  # All model inputs present.


def test_missing_dataset_raises_clear_error(tmp_path):  # Missing file must be actionable.
    with pytest.raises(FileNotFoundError, match="Kaggle"):  # Point users at the source.
        load_dataset(tmp_path / "missing.csv")  # Nonexistent path.


def test_labels_are_rule_separable():  # Dataset integrity check.
    """Verify the labels follow the documented rule (data-quality check)."""
    df = load_dataset()  # Load the real data.
    rule = (  # The exact generating condition.
        (df["cpu_utilization"] > 80)
        & (df["memory_usage"] > 85)
        & (df["temperature"] > 75)
    ).astype(int)  # 0/1 labels.
    assert (rule == df["system_status"]).all()  # Every row matches.


# --- API --------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):  # Isolated API client with fake models.
    monkeypatch.setattr(api_main, "DB_PATH", tmp_path / "test_predictions.db")  # Keep logs in tmp.
    monkeypatch.setattr(api_main, "load_models", lambda: None)  # Skip MLflow during startup.
    mock_lite, mock_heavy = MagicMock(), MagicMock()  # Deterministic model doubles.
    mock_lite.predict_proba.return_value = np.array([[0.90, 0.10]])  # Confident normal.
    mock_heavy.predict_proba.return_value = np.array([[0.10, 0.90]])  # Confident anomaly.
    with TestClient(api_main.app) as test_client:  # Run startup/shutdown hooks.
        api_main.models["fast_lite"] = mock_lite  # Inject the fast tier.
        api_main.models["heavy_pro"] = mock_heavy  # Inject the heavy tier.
        yield test_client  # Hand the client to the test.


def test_root_endpoint(client):  # Landing payload is discoverable.
    body = client.get("/").json()  # Call the base URL.
    assert body["service"] == "SentinelRoute AIOps API"  # Stable service name.
    assert body["health"] == "/health"  # Points at readiness.


def test_health_endpoint(client):  # Readiness contract.
    resp = client.get("/health")  # Call health.
    assert resp.status_code == 200  # Ready when both models exist.
    assert resp.json()["status"] == "ok"  # Expected status string.


def test_health_reports_model_alias(client):  # Deployment metadata.
    body = client.get("/health").json()  # Read the payload.
    assert body["alias"] == "champion"  # API resolves the champion alias.
    assert "model_versions" in body  # Versions are exposed.


def test_predict_valid_payload(client):  # Happy-path inference.
    resp = client.post("/predict", json=SAMPLE_PAYLOAD)  # Valid telemetry.
    assert resp.status_code == 200  # Request accepted.
    assert resp.headers["X-Model-Route"] == resp.json()["model_used"]  # Header matches body.


def test_predict_invalid_payload_422(client):  # Validation guard.
    bad = dict(SAMPLE_PAYLOAD)  # Copy a valid payload.
    bad["cpu_utilization"] = -5.0  # Break the range constraint.
    assert client.post("/predict", json=bad).status_code == 422  # FastAPI rejects it.


def test_stats_endpoint(client):  # Aggregation contract.
    client.post("/predict", json=SAMPLE_PAYLOAD)  # Generate one log row.
    assert client.get("/stats").json()["total_requests"] >= 1  # Counter moved.


# --- Cascade routing --------------------------------------------------------

@pytest.fixture
def cascade_client(tmp_path, monkeypatch):  # Expose the mocks to routing tests.
    monkeypatch.setattr(api_main, "DB_PATH", tmp_path / "cascade_test.db")  # Separate DB.
    monkeypatch.setattr(api_main, "load_models", lambda: None)  # No MLflow needed.
    mock_lite, mock_heavy = MagicMock(), MagicMock()  # Configurable doubles.
    with TestClient(api_main.app) as client:  # Trigger lifespan.
        api_main.models["fast_lite"] = mock_lite  # Inject fast tier.
        api_main.models["heavy_pro"] = mock_heavy  # Inject heavy tier.
        yield client, mock_lite, mock_heavy  # Give tests control of both tiers.


def test_fast_lite_confident_low(cascade_client):  # Low confidence stays on fast.
    client, mock_lite, mock_heavy = cascade_client  # Unpack the fixture.
    mock_lite.predict_proba.return_value = np.array([[0.90, 0.10]])  # p=0.10.
    resp = client.post("/predict", json=SAMPLE_PAYLOAD).json()  # Predict.
    assert resp["model_used"] == "fast-lite" and resp["cascade_triggered"] is False  # No escalation.
    assert resp["fast_probability"] == 0.10 and resp["heavy_probability"] is None  # Only fast scored.
    mock_heavy.predict_proba.assert_not_called()  # Heavy tier untouched.


def test_fast_lite_confident_high(cascade_client):  # High confidence stays on fast.
    client, mock_lite, mock_heavy = cascade_client  # Unpack the fixture.
    mock_lite.predict_proba.return_value = np.array([[0.15, 0.85]])  # p=0.85.
    resp = client.post("/predict", json=SAMPLE_PAYLOAD).json()  # Predict.
    assert resp["model_used"] == "fast-lite" and resp["prediction"] == 1  # Still an anomaly.
    assert resp["fast_probability"] == 0.85 and resp["heavy_probability"] is None  # Fast alone.
    mock_heavy.predict_proba.assert_not_called()  # Heavy tier untouched.


def test_uncertain_escalates_to_heavy(cascade_client):  # Uncertain band escalates.
    client, mock_lite, mock_heavy = cascade_client  # Unpack the fixture.
    mock_lite.predict_proba.return_value = np.array([[0.48, 0.52]])  # p=0.52, inside the band.
    mock_heavy.predict_proba.return_value = np.array([[0.18, 0.82]])  # Heavy confirms.
    resp = client.post("/predict", json=SAMPLE_PAYLOAD).json()  # Predict.
    assert resp["model_used"] == "heavy-pro" and resp["cascade_triggered"] is True  # Escalated.
    assert resp["fast_probability"] == 0.52 and resp["heavy_probability"] == 0.82  # Both reported.
    mock_heavy.predict_proba.assert_called_once()  # Heavy tier used exactly once.


# --- Drift and target service ----------------------------------------------

def _reference_features():  # Build a synthetic training baseline.
    rng = np.random.default_rng(0)  # Fixed seed for repeatability.
    return {  # One normal distribution per feature.
        col: {"sample_values": rng.normal(50.0, 5.0, 2000).tolist()}
        for col in FEATURE_COLS
    }


def test_stable_traffic_reports_no_drift():  # Same distribution means no drift.
    rng = np.random.default_rng(1)  # Independent sample.
    live = pd.DataFrame({col: rng.normal(50.0, 5.0, 2000) for col in FEATURE_COLS})  # Stable traffic.
    results = evaluate_drift(live, _reference_features())  # Run KS tests.
    drifted = [col for col, _, p in results if p < 0.05]  # Any false positives?
    assert drifted == []  # None expected.


def test_shifted_cpu_is_flagged():  # A real shift must be detected.
    rng = np.random.default_rng(2)  # Independent sample.
    live = pd.DataFrame({col: rng.normal(50.0, 5.0, 2000) for col in FEATURE_COLS})  # Stable baseline.
    live["cpu_utilization"] = rng.normal(85.0, 5.0, 2000)  # Shift one feature.
    results = evaluate_drift(live, _reference_features())  # Run KS tests.
    drifted = [col for col, _, p in results if p < 0.05]  # Collect significant features.
    assert drifted == ["cpu_utilization"]  # Only the shifted feature is flagged.


def test_target_fault_toggle():  # Fault endpoint drives healing demos.
    with TestClient(target_app) as tc:  # Start the target app.
        assert tc.get("/health").json()["fault_active"] is False  # Healthy by default.
        assert tc.post("/inject-fault").json()["fault_active"] is True  # Fault turns on.
        assert tc.get("/telemetry").status_code == 200  # Telemetry still served.
        assert tc.post("/clear-fault").json()["fault_active"] is False  # Fault turns off.

from src.data_prep import prepare_and_validate_batch
from src.quality_gate import evaluate_quality_gate


def test_batch_validation_detects_duplicates_and_distribution(tmp_path):
    batch_file = tmp_path / "batch.csv"
    df = pd.DataFrame([SAMPLE_PAYLOAD, SAMPLE_PAYLOAD])
    df["system_status"] = [0, 0]
    df.to_csv(batch_file, index=False)

    cleaned = prepare_and_validate_batch(batch_file)
    assert len(cleaned) == 1
    assert cleaned["system_status"].iloc[0] == 0


def test_quality_gate_promotes_and_retains(tmp_path):
    promoted = evaluate_quality_gate(
        champion_metrics={"f1_score": 0.88},
        candidate_metrics={"f1_score": 0.92},
        min_f1=0.85,
        output_path=tmp_path / "gate_pass.json",
    )
    assert promoted["eligible_for_promotion"] is True
    assert promoted["decision"] == "ELIGIBLE_FOR_PROMOTION"

    retained = evaluate_quality_gate(
        champion_metrics={"f1_score": 0.94},
        candidate_metrics={"f1_score": 0.89},
        min_f1=0.85,
        output_path=tmp_path / 
    )
