-- Phase 1 :: Indexing strategy
-- Rationale: the analytics views filter and group by time, department,
-- insurance provider, risk band and claim status, and join on the FK columns.
-- Idempotent.

SET search_path TO capstone_solution, public;

-- Foreign-key join columns (Postgres does not index FKs automatically)
CREATE INDEX IF NOT EXISTS ix_visits_patient_id  ON visits  (patient_id);
CREATE INDEX IF NOT EXISTS ix_billing_visit_id   ON billing  (visit_id);

-- Time-series slicing
CREATE INDEX IF NOT EXISTS ix_visits_visit_date    ON visits  (visit_date);
CREATE INDEX IF NOT EXISTS ix_billing_billing_date ON billing (billing_date);

-- Categorical group-by / filter columns
CREATE INDEX IF NOT EXISTS ix_visits_department   ON visits  (department);
CREATE INDEX IF NOT EXISTS ix_visits_risk_score   ON visits  (risk_score);
CREATE INDEX IF NOT EXISTS ix_visits_visit_type   ON visits  (visit_type);
CREATE INDEX IF NOT EXISTS ix_billing_claim_status ON billing (claim_status);
CREATE INDEX IF NOT EXISTS ix_patients_insurance_provider ON patients (insurance_provider);
CREATE INDEX IF NOT EXISTS ix_patients_city       ON patients (city);

-- Composite index supporting department-by-month performance rollups
CREATE INDEX IF NOT EXISTS ix_visits_dept_date ON visits (department, visit_date);

ANALYZE patients;
ANALYZE visits;
ANALYZE billing;
