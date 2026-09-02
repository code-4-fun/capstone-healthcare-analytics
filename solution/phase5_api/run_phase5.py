"""Phase 5 entrypoint - build the serving config, test, benchmark, report.

    uv run python phase5_api/run_phase5.py

Idempotent. Steps:

  1. ensure the Phase 3 model artefacts exist (rebuild Phase 3 if not)
  2. (re)derive the Model B operating threshold and write serving_config.json
  3. apply the prediction_log DDL
  4. run the pytest suite (schema / golden / contract / logging)
  5. boot the app in-process, exercise every endpoint, benchmark latency
  6. write charts, PHASE5_FINDINGS.md and openapi.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOLUTION = HERE.parent
sys.path.insert(0, str(SOLUTION / "src"))
sys.path.insert(0, str(HERE))

from capstone import modeling as M  # noqa: E402
from capstone import serving as S  # noqa: E402

import make_charts  # noqa: E402
import report  # noqa: E402

MANIFEST = S.PHASE3_MODELS_DIR / "training_manifest.json"
SERVING_CONFIG = HERE / "serving_config.json"
FINDINGS_MD = HERE / "PHASE5_FINDINGS.md"
N_BENCH = 500

ENDPOINTS = pd.DataFrame([
    ["GET", "/health", "Model + database readiness, uptime"],
    ["GET", "/model-info", "Versions, feature counts, thresholds, categorical domains"],
    ["POST", "/predict/claim-outcome", "Model B - pre-submission claim outcome + review/submit decision"],
    ["POST", "/predict/visit-risk", "Model A - visit risk (calibrated base-rate monitor)"],
], columns=["method", "route", "purpose"])


def step(n: int, msg: str) -> None:
    print(f"[{n}] {msg}")


def ensure_phase3() -> None:
    if MANIFEST.exists():
        step(1, f"Phase 3 artefacts present ({MANIFEST.relative_to(SOLUTION)})")
        return
    step(1, "Phase 3 artefacts missing; rebuilding via run_phase3.py")
    subprocess.run([sys.executable, str(SOLUTION / "phase3_models" / "run_phase3.py")], check=True)


def write_config() -> dict:
    step(2, "derive Model B operating threshold + write serving_config.json")
    S.write_serving_config(SERVING_CONFIG)
    cfg = json.loads(SERVING_CONFIG.read_text())
    b = cfg["models"]["B"]
    print(f"    threshold P(Rejected) >= {b['operating_threshold']} "
          f"({b.get('threshold_source', '?')}), threshold_version {b['threshold_version']}")
    return cfg


def apply_ddl() -> bool:
    step(3, "apply prediction_log DDL")
    from app import predictions_log as plog

    ok = plog.ensure_table()
    print(f"    prediction_log ready: {ok}")
    return ok


def run_tests() -> dict:
    step(4, "run pytest suite")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(HERE / "tests"), "-p", "no:cacheprovider",
         "-q", "--no-header", "-o", "addopts="],
        cwd=SOLUTION, capture_output=True, text=True,
    )
    import re

    summary = ""
    for line in reversed(proc.stdout.strip().splitlines()):
        if re.search(r"\d+ (passed|failed|error)", line):
            summary = line.strip("= ").strip()
            break
    print("   ", summary or proc.stdout.strip().splitlines()[-1])
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", summary)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", summary)) else 0
    errors = int(m.group(1)) if (m := re.search(r"(\d+) error", summary)) else 0
    if proc.returncode != 0 and failed == 0 and errors == 0:
        print(proc.stdout[-3000:])
        raise SystemExit("pytest failed")
    if failed or errors:
        print(proc.stdout[-3000:])
        raise SystemExit(f"pytest: {failed} failed, {errors} errors")
    return {"passed": passed, "failed": failed, "rc": proc.returncode, "summary": summary}


def _payloads(model: str, n: int) -> list[dict]:
    df = pd.read_parquet(S.PHASE2_PARQUET)
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    full_test = M.time_split(df).test
    test = full_test.sample(min(n, len(full_test)), random_state=7)
    rows = []
    for _, r in test.iterrows():
        p = {
            "visit_date": pd.Timestamp(r["visit_date"]).date().isoformat(),
            "department": r["department"], "visit_type": r["visit_type"],
            "age": int(r["age"]), "gender": r["gender"], "city": r["city"],
            "insurance_provider": r["insurance_provider"], "chronic_flag": bool(r["chronic_flag"]),
        }
        if model == "B":
            p["billed_amount"] = float(r["billed_amount"])
            p["length_of_stay_hours"] = float(r["length_of_stay_hours"])
            p["risk_score"] = r["risk_score"]
        rows.append(p)
    return rows


def benchmark(client) -> dict:
    step(5, f"benchmark ({N_BENCH} requests / endpoint) + exercise endpoints")
    out: dict = {"latency_samples": {}, "rows": [], "claim_mix": {}, "decision_split": {}, "risk_mix": {}}

    for model, endpoint in (("B", "/predict/claim-outcome"), ("A", "/predict/visit-risk")):
        payloads = _payloads(model, N_BENCH)
        lat = np.empty(len(payloads))
        classes: list[str] = []
        actions: list[str] = []
        t0 = time.perf_counter()
        for i, pl in enumerate(payloads):
            resp = client.post(endpoint, json=pl)
            assert resp.status_code == 200, resp.text
            body = resp.json()
            lat[i] = body["latency_ms"]
            classes.append(body["predicted_class"])
            if model == "B":
                actions.append(body["decision"]["action"])
        wall = time.perf_counter() - t0
        out["latency_samples"][endpoint] = lat
        out["rows"].append({
            "endpoint": endpoint, "n": len(payloads),
            "p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
            "p99": float(np.percentile(lat, 99)), "mean": float(lat.mean()),
            "throughput_rps": float(len(payloads) / wall),
        })
        if model == "B":
            out["claim_mix"] = pd.Series(classes).value_counts()
            out["decision_split"] = pd.Series(actions).value_counts()
        else:
            out["risk_mix"] = pd.Series(classes).value_counts()

    out["latency"] = pd.DataFrame(out["rows"])
    return out


def golden_check(client) -> dict:
    g_dir = HERE / "tests" / "golden"
    res = {}
    for k, endpoint, tol in (("A", "/predict/visit-risk", 1e-6), ("B", "/predict/claim-outcome", 1e-6)):
        cases = json.loads((g_dir / f"model_{k.lower()}_golden.json").read_text())["cases"]
        label_mis = 0
        max_delta = 0.0
        for case in cases:
            body = client.post(endpoint, json=case["payload"]).json()
            if body["predicted_class"] != case["expected"]["predicted_class"]:
                label_mis += 1
            for cls, p in case["expected"]["probabilities"].items():
                max_delta = max(max_delta, abs(body["probabilities"][cls] - p))
        res[k] = {"n": len(cases), "label_mismatches": label_mis, "max_prob_delta": round(max_delta, 9)}
    return res


def sample_log_row(request_id: str) -> dict | None:
    try:
        from capstone.db import connect

        with connect(autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT endpoint, model, model_version, feature_spec_version, serving_version, "
                "operating_threshold, predicted_class, latency_ms "
                "FROM capstone_solution.prediction_log WHERE request_id = %s", (request_id,))
            row = cur.fetchone()
            if not row:
                return None
            from decimal import Decimal

            cols = ["endpoint", "model", "model_version", "feature_spec_version", "serving_version",
                    "operating_threshold", "predicted_class", "latency_ms"]
            return {c: (float(v) if isinstance(v, Decimal) else v) for c, v in zip(cols, row)}
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ensure_phase3()
    cfg = write_config()
    db_ready = apply_ddl()
    tests = run_tests()

    from fastapi.testclient import TestClient

    from app.main import create_app

    manifest = json.loads(MANIFEST.read_text())
    with TestClient(create_app()) as client:
        bench = benchmark(client)
        golden = golden_check(client)
        # one tagged prediction whose log row we quote in the findings
        probe = client.post("/predict/claim-outcome", json=_payloads("B", 1)[0]).json()
        step(6, "charts, PHASE5_FINDINGS.md, openapi.json")
        openapi = client.get("/openapi.json").json()

    (HERE / "openapi.json").write_text(json.dumps(openapi, indent=2) + "\n")

    ctx = report.Phase5Context(
        config=cfg, manifest=manifest, endpoints=ENDPOINTS, tests=tests, golden=golden,
        latency=bench["latency"], latency_samples=bench["latency_samples"],
        claim_mix=bench["claim_mix"], decision_split=bench["decision_split"], risk_mix=bench["risk_mix"],
        sample_log_row=sample_log_row(probe["request_id"]),
        db_logged=bool(db_ready and sample_log_row(probe["request_id"]) is not None),
    )
    charts = make_charts.build_all(ctx)
    report.write_findings(ctx, charts, FINDINGS_MD)

    print(f"\nDone.\n  findings : {FINDINGS_MD.relative_to(SOLUTION)}")
    print(f"  charts   : {', '.join(k for k, _, _ in charts)}")
    print(f"  openapi  : {(HERE / 'openapi.json').relative_to(SOLUTION)}")
    for r in bench["rows"]:
        print(f"  {r['endpoint']:<26} p50 {r['p50']:.2f} ms  p95 {r['p95']:.2f} ms  "
              f"p99 {r['p99']:.2f} ms  ({r['throughput_rps']:.0f} rps)")


if __name__ == "__main__":
    main()
