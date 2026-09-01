-- Phase 1 :: Core typed tables (direct typed load target)
-- Relational design with integrity enforced at the database level:
--   * primary keys on every entity
--   * foreign keys wiring visits -> patients and billing -> visits
--   * CHECK constraints encoding the known business domains
--   * billing is 1:1 with visits (enforced by UNIQUE on visit_id)
-- Idempotent: safe to re-run (drops and recreates).

SET search_path TO capstone_solution, public;

DROP TABLE IF EXISTS billing CASCADE;
DROP TABLE IF EXISTS visits CASCADE;
DROP TABLE IF EXISTS patients CASCADE;
DROP TABLE IF EXISTS load_rejects CASCADE;

-- ---------------------------------------------------------------------------
-- patients : one row per registered patient
-- ---------------------------------------------------------------------------
CREATE TABLE patients (
    patient_id         INTEGER      PRIMARY KEY,
    age                SMALLINT     NOT NULL CHECK (age BETWEEN 0 AND 120),
    gender             CHAR(1)      NOT NULL CHECK (gender IN ('M', 'F')),
    city               TEXT         NOT NULL,
    insurance_provider TEXT         NOT NULL,
    chronic_flag       SMALLINT     NOT NULL CHECK (chronic_flag IN (0, 1)),
    registration_date  DATE         NOT NULL
);

COMMENT ON TABLE  patients IS 'Registered patients across the multi-city hospital network';
COMMENT ON COLUMN patients.chronic_flag IS '1 = patient has a known chronic condition';
COMMENT ON COLUMN patients.registration_date IS 'Date the patient record was first created (see data-quality notes: not a reliable temporal anchor)';

-- ---------------------------------------------------------------------------
-- visits : one row per hospital visit / encounter
-- ---------------------------------------------------------------------------
CREATE TABLE visits (
    visit_id             INTEGER      PRIMARY KEY,
    patient_id           INTEGER      NOT NULL REFERENCES patients (patient_id),
    visit_date           DATE         NOT NULL,
    department           TEXT         NOT NULL,
    visit_type           TEXT         NOT NULL CHECK (visit_type IN ('ER', 'OPD', 'ICU')),
    length_of_stay_hours NUMERIC(6,2) NOT NULL CHECK (length_of_stay_hours >= 0),
    risk_score           TEXT         NOT NULL CHECK (risk_score IN ('Low', 'Medium', 'High')),
    doctor_id            INTEGER      NOT NULL
);

COMMENT ON TABLE  visits IS 'Hospital visits / encounters; grain = one visit';
COMMENT ON COLUMN visits.risk_score IS 'Operational-clinical risk band assigned at the visit (target for Phase 3 Model A)';
COMMENT ON COLUMN visits.length_of_stay_hours IS 'Length of stay in hours; observed floor of 0.5h looks like a measurement/clip artefact';

-- ---------------------------------------------------------------------------
-- billing : one row per claim; 1:1 with visits
-- ---------------------------------------------------------------------------
CREATE TABLE billing (
    bill_id         INTEGER       PRIMARY KEY,
    visit_id        INTEGER       NOT NULL UNIQUE REFERENCES visits (visit_id),
    billed_amount   NUMERIC(12,2) NOT NULL CHECK (billed_amount >= 0),
    approved_amount NUMERIC(12,2)          CHECK (approved_amount >= 0),
    claim_status    TEXT          NOT NULL CHECK (claim_status IN ('Paid', 'Pending', 'Rejected')),
    payment_days    NUMERIC(6,2)           CHECK (payment_days >= 0),
    billing_date    DATE          NOT NULL,
    CONSTRAINT billing_approved_not_over_billed
        CHECK (approved_amount IS NULL OR approved_amount <= billed_amount)
);

COMMENT ON TABLE  billing IS 'Insurance claim / billing record per visit; grain = one visit';
COMMENT ON COLUMN billing.approved_amount IS 'Amount approved by the insurer; NULL when not yet adjudicated';
COMMENT ON COLUMN billing.payment_days IS 'Days from billing to payment; NULL when unpaid (typically Pending/Rejected)';
COMMENT ON COLUMN billing.claim_status IS 'Claim outcome (target for Phase 3 Model B)';

-- ---------------------------------------------------------------------------
-- load_rejects : rows the typed load could not accept, captured for audit
-- ---------------------------------------------------------------------------
CREATE TABLE load_rejects (
    reject_id    BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    loaded_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_table TEXT        NOT NULL,
    source_row   JSONB       NOT NULL,
    reason       TEXT        NOT NULL
);

COMMENT ON TABLE load_rejects IS 'Audit trail of source rows rejected during the Phase 1 typed load';
