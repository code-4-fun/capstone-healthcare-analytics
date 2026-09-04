# Incident runbook - monitoring & serving

On-call: Analytics & AI team. Escalation: Revenue-cycle lead (Model B impact),
Data platform / DBA (database). All timestamps in the platform are UTC.

Dashboards: Grafana **Hospital Drift** (`http://<host>:3000`). Tables:
`capstone_solution.drift_report`, `prediction_log`, `v_prediction_audit`.

---

## 1. Drift alert fired (`drift_report.alert = true`)

**Detect:** Grafana "Active drift alerts" panel is non-empty, or
`SELECT * FROM capstone_solution.drift_report WHERE alert AND run_ts > now() - interval '1 day'`.

**Triage:**
1. Identify the `metric_kind` and `feature`.
   - `feature_psi` on `billed_amount` / `billed_band` -> a tariff or
     case-mix change. Confirm with revenue-cycle.
   - `feature_psi` on `department` / `city` / `insurance_provider` -> a new
     site, service line or payer contract. Confirm with operations.
   - `prediction_psi` -> the output mix moved; usually downstream of a
     `feature_psi` alert on the same run.
   - `perf_recall_costly` -> Model B is missing more rejections. **This is the
     one with revenue impact.**
   - `gate_fail_rate` -> malformed requests; go to section 2.
2. Check whether it is **one run or two consecutive**. A single-run PSI spike on
   a low-volume window is often noise - wait for the next scheduled run.
3. Check the window size (`window_n`). Below ~200 requests, treat PSI cautiously.

**Act:**
- Single transient spike -> annotate the Grafana panel, no action.
- Sustained feature/prediction drift (trigger 1 or 2 in `retraining_policy.md`)
  -> open a retraining ticket; keep serving (the model still returns calibrated
  probabilities, just against a shifted population).
- `perf_recall_costly` below 0.52 on a trailing quarter -> **page the
  revenue-cycle lead**, open a priority retraining ticket, and consider
  temporarily lowering the operating threshold (more review alerts) as a
  stopgap - this is a `serving_config.json` change, not a model change.

---

## 2. Validation-gate failure rate spiked

**Detect:** `gate_fail_rate` alert, or Grafana "Validation-gate fail rate" panel
red.

**Act:**
1. `SELECT detail FROM capstone_solution.drift_report WHERE metric_kind = 'gate_fail_rate' ORDER BY run_ts DESC LIMIT 1`
   - the `by_rule` breakdown names the offending rules and `offending` lists
   request ids.
2. If it is one rule (e.g. `department_enum`) -> a caller is sending a new valid
   value the platform does not know yet. Confirm it is legitimate, then add it
   to the domain in `capstone.data_quality` **and** `capstone.serving.DOMAINS`,
   and plan a retrain so the model has seen it.
3. If it is many rules from one `client_host` -> a broken integration. Contact
   that caller; predictions for malformed requests are still served (the gate is
   a monitor, not an admission control) but should be treated as unreliable.

---

## 3. API latency regression

**Detect:** external APM, or `SELECT percentile_disc(0.95) WITHIN GROUP (ORDER BY latency_ms)
FROM capstone_solution.prediction_log WHERE ts > now() - interval '1 hour'`
above ~60 ms (baseline: claim-outcome ~22 ms, visit-risk ~6 ms compute-only).

**Act:**
1. Check host CPU / memory and Postgres health (the log insert is best-effort
   and off the response path, but a saturated host slows scoring).
2. Restart the API container. Model load happens once at startup; a restart is
   ~20 s.
3. If latency is fine but throughput dropped -> check the upstream caller.

---

## 4. Model rollback

Use when a promoted model or threshold is worse in production than the
incumbent (see `retraining_policy.md` section 3).

1. Restore the previous `phase3_models/models/` directory and
   `phase5_api/serving_config.json` from the retained version.
2. Restart the API: `docker compose -f phase5_api/docker-compose.yml up -d --build api`
   (or the Phase 6 compose).
3. Verify: `GET /model-info` shows the rolled-back `model_version`; a probe
   prediction logs with that version.
4. The drift reference reverts with the model (it is keyed to the Phase 3 test
   window of that version). Note the rollback in the `retraining_policy.md`
   change log.

---

## 5. Prediction log unavailable

**Symptom:** `GET /health` returns `db_reachable: false`; predictions still
return `200`.

**Impact:** predictions are **still served** (the log write is best-effort and
never fails a request). The gap means **no drift monitoring** for that period
and a hole in the audit trail.

**Act:**
1. Bring Postgres back (DBA). The API needs no restart - the next log write
   reconnects.
2. Once back, run a catch-up drift job for the covered window:
   `uv run python -m phase6_monitoring drift-job --window "<start>..<end>"`.
   Requests during the outage are simply absent from the reference set - record
   the gap in the incident log.
3. If the outage exceeded the retention/monitoring lookback, note in
   `governance.md`'s review that a monitoring blind spot occurred.

---

## 6. Scheduler not producing drift runs

**Detect:** no new `drift_report.run_id` in the expected interval.

**Act:**
1. `docker compose -f phase6_monitoring/docker-compose.yml logs scheduler` -
   look for the "drift run failed" line and its traceback.
2. Common causes: Postgres unreachable from the container (check `PGHOST`), or
   the serving artefacts missing from the image (rebuild).
3. Run one pass by hand to confirm the fix:
   `docker compose -f phase6_monitoring/docker-compose.yml exec scheduler python -m phase6_monitoring drift-job`.
