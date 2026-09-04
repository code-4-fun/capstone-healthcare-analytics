-- Phase 6 :: drift report + governance audit
--
-- The Phase 5 prediction_log is the monitored signal. This script adds:
--   * drift_report        - one row per (drift run, metric); written by the
--                           scheduled drift job (phase6_monitoring/drift_job.py)
--   * prediction_override - the who / what / when of any manual override of a
--                           served prediction
--   * v_prediction_audit  - every prediction joined to any override: the
--                           governance audit trail
--   * forbid_mutation()   - trigger enforcing immutability. A served prediction
--                           is never edited (UPDATE blocked on all three logs);
--                           the audit tables are never pruned either (DELETE
--                           also blocked on drift_report + prediction_override).
--                           prediction_log keeps DELETE for retention / cleanup
--                           - the retention policy is in governance.md.
--
-- Idempotent - safe to re-run.

CREATE SCHEMA IF NOT EXISTS capstone_solution;

-- --------------------------------------------------------------------------
-- drift_report
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS capstone_solution.drift_report (
    id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_ts           timestamptz NOT NULL DEFAULT now(),
    run_id           uuid        NOT NULL,
    model            text        NOT NULL CHECK (model IN ('A', 'B')),
    window_start     timestamptz,
    window_end       timestamptz,
    window_n         integer     NOT NULL,
    -- feature_psi | feature_ks | prediction_psi | class_share
    -- | perf_recall_costly | perf_net_recovered | gate_fail_rate
    metric_kind      text        NOT NULL,
    feature          text,                         -- null for non-feature metrics
    value            numeric,
    reference        numeric,                      -- the baseline this metric is judged against
    band             text,                         -- stable|moderate|significant|ok|warn|alert|info
    alert            boolean     NOT NULL DEFAULT false,
    detail           jsonb,
    model_version    text        NOT NULL,
    reference_window text        NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_drift_report_model_ts
    ON capstone_solution.drift_report (model, run_ts DESC);
CREATE INDEX IF NOT EXISTS ix_drift_report_run
    ON capstone_solution.drift_report (run_id);
CREATE INDEX IF NOT EXISTS ix_drift_report_alert
    ON capstone_solution.drift_report (run_ts DESC) WHERE alert;

COMMENT ON TABLE capstone_solution.drift_report IS
    'Phase 6 drift monitor output - one row per (run, metric). Append-only.';

-- --------------------------------------------------------------------------
-- prediction_override  (manual overrides - the "who / what / when")
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS capstone_solution.prediction_override (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts             timestamptz NOT NULL DEFAULT now(),
    request_id     uuid        NOT NULL,
    model          text        NOT NULL CHECK (model IN ('A', 'B')),
    original_class text        NOT NULL,
    override_class text        NOT NULL,
    actor          text        NOT NULL,           -- who made the call
    reason         text        NOT NULL,           -- why
    source         text        NOT NULL DEFAULT 'manual'
);

CREATE INDEX IF NOT EXISTS ix_prediction_override_request
    ON capstone_solution.prediction_override (request_id);

COMMENT ON TABLE capstone_solution.prediction_override IS
    'Manual overrides of served predictions - governance audit. Append-only.';

-- --------------------------------------------------------------------------
-- v_prediction_audit  (every prediction + any override)
-- --------------------------------------------------------------------------
CREATE OR REPLACE VIEW capstone_solution.v_prediction_audit AS
SELECT
    p.request_id,
    p.ts                         AS predicted_at,
    p.model,
    p.model_version,
    p.feature_spec_version,
    p.serving_version,
    p.endpoint,
    p.predicted_class,
    p.probabilities,
    p.decision,
    p.operating_threshold,
    p.client_host,
    o.ts                         AS overridden_at,
    o.override_class,
    o.actor                      AS override_actor,
    o.reason                     AS override_reason,
    (o.id IS NOT NULL)           AS was_overridden
FROM capstone_solution.prediction_log p
LEFT JOIN capstone_solution.prediction_override o USING (request_id);

COMMENT ON VIEW capstone_solution.v_prediction_audit IS
    'Governance audit trail: every prediction with its versions and any manual override.';

-- --------------------------------------------------------------------------
-- append-only enforcement (immutable-by-convention, actually enforced)
-- --------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION capstone_solution.forbid_mutation()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'capstone_solution.% is immutable (attempted %)',
        TG_TABLE_NAME, TG_OP
        USING HINT = 'insert a correcting row instead of updating or deleting';
END;
$$;

-- UPDATE is blocked everywhere (a logged prediction / drift metric / override is
-- never edited). DELETE is additionally blocked on the audit tables; it stays
-- open on prediction_log for retention pruning and test / seed cleanup.
DO $$
DECLARE spec record;
BEGIN
    FOR spec IN
        SELECT * FROM (VALUES
            ('prediction_log',      'UPDATE'),
            ('drift_report',        'UPDATE OR DELETE'),
            ('prediction_override', 'UPDATE OR DELETE')
        ) AS s(tbl, ops)
    LOOP
        IF to_regclass('capstone_solution.' || spec.tbl) IS NOT NULL THEN
            EXECUTE format('DROP TRIGGER IF EXISTS %I_immutable ON capstone_solution.%I',
                           spec.tbl, spec.tbl);
            EXECUTE format(
                'CREATE TRIGGER %I_immutable BEFORE %s ON capstone_solution.%I '
                'FOR EACH ROW EXECUTE FUNCTION capstone_solution.forbid_mutation()',
                spec.tbl, spec.ops, spec.tbl);
        END IF;
    END LOOP;
END;
$$;
