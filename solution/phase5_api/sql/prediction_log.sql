-- Phase 5 :: prediction log
--
-- Every prediction served by the Phase 5 API is appended here with its model +
-- version metadata, latency and the request payload. This table is the drift
-- baseline Phase 6 monitors (feature drift, prediction-distribution drift, and -
-- once outcomes land - performance drift), and the audit trail for governance.
--
-- Append-only by convention: the API only ever INSERTs. Idempotent - safe to
-- re-run.

CREATE SCHEMA IF NOT EXISTS capstone_solution;

CREATE TABLE IF NOT EXISTS capstone_solution.prediction_log (
    id                    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts                    timestamptz  NOT NULL DEFAULT now(),
    request_id            uuid         NOT NULL,
    endpoint              text         NOT NULL,
    model                 text         NOT NULL CHECK (model IN ('A', 'B')),
    model_version         text         NOT NULL,
    feature_spec_version  integer      NOT NULL,
    serving_version       text         NOT NULL,
    operating_threshold   numeric,
    predicted_class       text         NOT NULL,
    probabilities         jsonb        NOT NULL,
    decision              jsonb,
    defaults_applied      jsonb        NOT NULL DEFAULT '[]'::jsonb,
    request_payload       jsonb,
    latency_ms            numeric      NOT NULL,
    client_host           text
);

CREATE INDEX IF NOT EXISTS ix_prediction_log_model_ts
    ON capstone_solution.prediction_log (model, ts DESC);

CREATE INDEX IF NOT EXISTS ix_prediction_log_request_id
    ON capstone_solution.prediction_log (request_id);

COMMENT ON TABLE capstone_solution.prediction_log IS
    'Phase 5 API prediction log - drift baseline + audit trail for Phase 6.';
