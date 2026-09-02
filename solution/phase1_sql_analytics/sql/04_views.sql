-- Phase 1 :: Business intelligence views
-- A trusted, reusable analytics layer over the core tables. Downstream phases
-- (EDA, modelling) should read from v_visit_billing rather than re-joining.
-- All views are CREATE OR REPLACE (idempotent).

SET search_path TO capstone_solution, public;

-- ===========================================================================
-- v_visit_billing : enriched fact spine (one row per visit)
-- ===========================================================================
CREATE OR REPLACE VIEW v_visit_billing AS
SELECT
    v.visit_id,
    v.visit_date,
    date_trunc('month', v.visit_date)::date            AS visit_month,
    v.patient_id,
    p.age,
    CASE
        WHEN p.age < 18 THEN '0-17'
        WHEN p.age < 35 THEN '18-34'
        WHEN p.age < 50 THEN '35-49'
        WHEN p.age < 65 THEN '50-64'
        ELSE '65+'
    END                                                AS age_band,
    p.gender,
    p.city,
    p.insurance_provider,
    p.chronic_flag,
    p.registration_date,
    v.department,
    v.visit_type,
    v.length_of_stay_hours,
    v.risk_score,
    (v.risk_score = 'High')                            AS is_high_risk,
    v.doctor_id,
    b.bill_id,
    b.billing_date,
    date_trunc('month', b.billing_date)::date          AS billing_month,
    b.billed_amount,
    b.approved_amount,
    b.claim_status,
    b.payment_days,
    (b.claim_status = 'Paid')                          AS is_paid,
    (b.claim_status = 'Pending')                       AS is_pending,
    (b.claim_status = 'Rejected')                      AS is_rejected,
    -- Cash actually collected: approved value on paid claims only
    CASE WHEN b.claim_status = 'Paid'
         THEN COALESCE(b.approved_amount, 0) ELSE 0 END AS collected_amount,
    -- Revenue leakage: billed value not approved once the claim is adjudicated
    CASE WHEN b.claim_status IN ('Paid', 'Rejected')
         THEN b.billed_amount - COALESCE(b.approved_amount, 0) ELSE 0 END AS leakage_amount,
    -- Revenue still at risk (awaiting adjudication)
    CASE WHEN b.claim_status = 'Pending' THEN b.billed_amount ELSE 0 END  AS pending_amount,
    (b.billing_date - v.visit_date)                    AS billing_lag_days
FROM visits   v
JOIN patients p ON p.patient_id = v.patient_id
JOIN billing  b ON b.visit_id  = v.visit_id;

COMMENT ON VIEW v_visit_billing IS
    'Enriched one-row-per-visit fact spine (visit + patient + billing) reused by all downstream phases';

-- ===========================================================================
-- v_department_performance : operational + financial scorecard per department
-- ===========================================================================
CREATE OR REPLACE VIEW v_department_performance AS
SELECT
    department,
    COUNT(*)                                                       AS visits,
    COUNT(DISTINCT patient_id)                                     AS unique_patients,
    ROUND(AVG(length_of_stay_hours), 2)                            AS avg_los_hours,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY length_of_stay_hours)::numeric, 2)
                                                                  AS median_los_hours,
    ROUND(AVG(length_of_stay_hours) FILTER (WHERE visit_type = 'ICU'), 2)
                                                                  AS avg_los_hours_icu,
    COUNT(*) FILTER (WHERE risk_score = 'High')                    AS high_risk_visits,
    ROUND(100.0 * COUNT(*) FILTER (WHERE risk_score = 'High') / COUNT(*), 1)
                                                                  AS high_risk_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE visit_type = 'ER')  / COUNT(*), 1) AS er_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE visit_type = 'ICU') / COUNT(*), 1) AS icu_pct,
    SUM(billed_amount)                                             AS total_billed,
    SUM(collected_amount)                                          AS total_collected,
    SUM(leakage_amount)                                            AS total_leakage,
    SUM(pending_amount)                                            AS total_pending,
    ROUND(100.0 * SUM(collected_amount) / NULLIF(SUM(billed_amount), 0), 1) AS realization_rate_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE claim_status = 'Rejected') / COUNT(*), 1) AS rejection_rate_pct,
    ROUND(AVG(payment_days) FILTER (WHERE claim_status = 'Paid'), 1) AS avg_payment_days
FROM v_visit_billing
GROUP BY department
ORDER BY total_billed DESC;

COMMENT ON VIEW v_department_performance IS
    'Per-department operational (volume, LOS, risk, acuity mix) and financial (billed, collected, leakage, realization) scorecard';

-- ===========================================================================
-- v_patient_flow_daily : daily patient-flow trend
-- ===========================================================================
CREATE OR REPLACE VIEW v_patient_flow_daily AS
SELECT
    visit_date,
    COUNT(*)                                        AS visits,
    COUNT(*) FILTER (WHERE visit_type = 'ER')       AS er_visits,
    COUNT(*) FILTER (WHERE visit_type = 'OPD')      AS opd_visits,
    COUNT(*) FILTER (WHERE visit_type = 'ICU')      AS icu_visits,
    COUNT(*) FILTER (WHERE risk_score = 'High')     AS high_risk_visits,
    COUNT(DISTINCT patient_id)                      AS unique_patients,
    ROUND(AVG(length_of_stay_hours), 2)             AS avg_los_hours
FROM v_visit_billing
GROUP BY visit_date
ORDER BY visit_date;

COMMENT ON VIEW v_patient_flow_daily IS 'Daily patient-flow volume, acuity mix and average length of stay';

-- ===========================================================================
-- v_patient_flow_monthly : monthly flow by department
-- ===========================================================================
CREATE OR REPLACE VIEW v_patient_flow_monthly AS
SELECT
    visit_month,
    department,
    COUNT(*)                                        AS visits,
    COUNT(DISTINCT patient_id)                      AS unique_patients,
    COUNT(*) FILTER (WHERE risk_score = 'High')     AS high_risk_visits,
    ROUND(AVG(length_of_stay_hours), 2)             AS avg_los_hours,
    SUM(billed_amount)                              AS total_billed,
    SUM(collected_amount)                           AS total_collected
FROM v_visit_billing
GROUP BY visit_month, department
ORDER BY visit_month, department;

COMMENT ON VIEW v_patient_flow_monthly IS 'Monthly patient flow and revenue by department';

-- ===========================================================================
-- v_insurance_provider_behavior : how each insurer adjudicates and pays
-- ===========================================================================
CREATE OR REPLACE VIEW v_insurance_provider_behavior AS
SELECT
    insurance_provider,
    COUNT(*)                                                          AS claims,
    COUNT(*) FILTER (WHERE claim_status = 'Paid')                     AS paid_claims,
    COUNT(*) FILTER (WHERE claim_status = 'Pending')                  AS pending_claims,
    COUNT(*) FILTER (WHERE claim_status = 'Rejected')                 AS rejected_claims,
    ROUND(100.0 * COUNT(*) FILTER (WHERE claim_status = 'Paid')     / COUNT(*), 1) AS paid_rate_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE claim_status = 'Pending')  / COUNT(*), 1) AS pending_rate_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE claim_status = 'Rejected') / COUNT(*), 1) AS rejection_rate_pct,
    ROUND(AVG(approved_amount / NULLIF(billed_amount, 0))
          FILTER (WHERE claim_status IN ('Paid', 'Rejected')) * 100.0, 1)          AS avg_approved_ratio_pct,
    ROUND(AVG(payment_days) FILTER (WHERE claim_status = 'Paid'), 1)  AS avg_payment_days,
    ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP (
              ORDER BY CASE WHEN claim_status = 'Paid' THEN payment_days END)::numeric, 1)
                                                                     AS p90_payment_days,
    SUM(billed_amount)                                                AS total_billed,
    SUM(collected_amount)                                             AS total_collected,
    SUM(leakage_amount)                                               AS total_leakage,
    ROUND(100.0 * SUM(collected_amount) / NULLIF(SUM(billed_amount), 0), 1) AS realization_rate_pct
FROM v_visit_billing
GROUP BY insurance_provider
ORDER BY total_billed DESC;

COMMENT ON VIEW v_insurance_provider_behavior IS
    'Per-insurer claim outcome mix, approval ratio, payment speed and revenue realization';

-- ===========================================================================
-- v_revenue_realization_monthly : revenue waterfall by billing month
-- ===========================================================================
CREATE OR REPLACE VIEW v_revenue_realization_monthly AS
SELECT
    billing_month,
    COUNT(*)                                                  AS claims,
    SUM(billed_amount)                                        AS billed_amount,
    SUM(COALESCE(approved_amount, 0))                         AS approved_amount,
    SUM(collected_amount)                                     AS collected_amount,
    SUM(pending_amount)                                       AS pending_amount,
    SUM(CASE WHEN claim_status = 'Rejected' THEN billed_amount ELSE 0 END) AS rejected_amount,
    SUM(leakage_amount)                                       AS leakage_amount,
    ROUND(100.0 * SUM(collected_amount) / NULLIF(SUM(billed_amount), 0), 1) AS realization_rate_pct,
    ROUND(100.0 * SUM(leakage_amount)   / NULLIF(SUM(billed_amount), 0), 1) AS leakage_rate_pct
FROM v_visit_billing
GROUP BY billing_month
ORDER BY billing_month;

COMMENT ON VIEW v_revenue_realization_monthly IS
    'Monthly revenue waterfall: billed -> approved -> collected, with pending and leakage';

-- ===========================================================================
-- v_high_risk_visits : monitoring list of High-risk visits with full context
-- ===========================================================================
CREATE OR REPLACE VIEW v_high_risk_visits AS
SELECT
    visit_id, visit_date, department, visit_type, length_of_stay_hours,
    patient_id, age, age_band, gender, city, insurance_provider, chronic_flag,
    billed_amount, approved_amount, claim_status, payment_days
FROM v_visit_billing
WHERE risk_score = 'High'
ORDER BY visit_date DESC;

COMMENT ON VIEW v_high_risk_visits IS 'All High-risk visits with patient and billing context for operational review';

-- ===========================================================================
-- v_claim_rejection_analysis : where rejections concentrate
-- ===========================================================================
CREATE OR REPLACE VIEW v_claim_rejection_analysis AS
WITH base AS (
    SELECT
        department,
        insurance_provider,
        visit_type,
        risk_score,
        CASE
            WHEN billed_amount < 5000  THEN '1. <5k'
            WHEN billed_amount < 15000 THEN '2. 5k-15k'
            WHEN billed_amount < 30000 THEN '3. 15k-30k'
            ELSE '4. 30k+'
        END AS billed_band,
        is_rejected
    FROM v_visit_billing
)
SELECT 'department'         AS dimension, department         AS value,
       COUNT(*) AS claims, COUNT(*) FILTER (WHERE is_rejected) AS rejected,
       ROUND(100.0 * COUNT(*) FILTER (WHERE is_rejected) / COUNT(*), 1) AS rejection_rate_pct
FROM base GROUP BY department
UNION ALL
SELECT 'insurance_provider', insurance_provider,
       COUNT(*), COUNT(*) FILTER (WHERE is_rejected),
       ROUND(100.0 * COUNT(*) FILTER (WHERE is_rejected) / COUNT(*), 1)
FROM base GROUP BY insurance_provider
UNION ALL
SELECT 'visit_type', visit_type,
       COUNT(*), COUNT(*) FILTER (WHERE is_rejected),
       ROUND(100.0 * COUNT(*) FILTER (WHERE is_rejected) / COUNT(*), 1)
FROM base GROUP BY visit_type
UNION ALL
SELECT 'risk_score', risk_score,
       COUNT(*), COUNT(*) FILTER (WHERE is_rejected),
       ROUND(100.0 * COUNT(*) FILTER (WHERE is_rejected) / COUNT(*), 1)
FROM base GROUP BY risk_score
UNION ALL
SELECT 'billed_band', billed_band,
       COUNT(*), COUNT(*) FILTER (WHERE is_rejected),
       ROUND(100.0 * COUNT(*) FILTER (WHERE is_rejected) / COUNT(*), 1)
FROM base GROUP BY billed_band
ORDER BY dimension, rejection_rate_pct DESC;

COMMENT ON VIEW v_claim_rejection_analysis IS
    'Claim rejection rate sliced by department, insurer, visit type, risk band and billed-amount band';

-- ===========================================================================
-- v_doctor_workload : load and outcomes per doctor
-- ===========================================================================
CREATE OR REPLACE VIEW v_doctor_workload AS
SELECT
    doctor_id,
    COUNT(*)                                       AS visits,
    COUNT(DISTINCT patient_id)                     AS unique_patients,
    COUNT(DISTINCT department)                     AS departments_covered,
    COUNT(*) FILTER (WHERE risk_score = 'High')    AS high_risk_visits,
    ROUND(AVG(length_of_stay_hours), 2)            AS avg_los_hours,
    SUM(billed_amount)                             AS total_billed,
    ROUND(100.0 * COUNT(*) FILTER (WHERE claim_status = 'Rejected') / COUNT(*), 1) AS rejection_rate_pct
FROM v_visit_billing
GROUP BY doctor_id
ORDER BY visits DESC;

COMMENT ON VIEW v_doctor_workload IS 'Per-doctor visit load, acuity mix and billing outcomes';
