-- Phase 1 :: Data quality report
-- One row per automated check. Feeds the Phase 2 EDA and the Phase 6 governance
-- layer. Severity: ERROR (violates a hard business rule), WARN (suspicious /
-- likely artefact), INFO (context worth tracking).

SET search_path TO capstone_solution, public;

CREATE OR REPLACE VIEW v_data_quality_report AS
WITH checks AS (
    -- Referential completeness -------------------------------------------------
    SELECT 'patients_without_visits' AS check_name, 'INFO' AS severity,
           'patients' AS table_scope,
           (SELECT COUNT(*) FROM patients p
              WHERE NOT EXISTS (SELECT 1 FROM visits v WHERE v.patient_id = p.patient_id)) AS records_flagged,
           (SELECT COUNT(*) FROM patients) AS records_total,
           'Registered patients with zero visits' AS description

    UNION ALL
    SELECT 'visits_without_billing', 'ERROR', 'visits',
           (SELECT COUNT(*) FROM visits v
              WHERE NOT EXISTS (SELECT 1 FROM billing b WHERE b.visit_id = v.visit_id)),
           (SELECT COUNT(*) FROM visits),
           'Visits with no billing record (expected 1:1)'

    -- Temporal consistency --------------------------------------------------
    UNION ALL
    SELECT 'visit_before_registration', 'WARN', 'visits + patients',
           (SELECT COUNT(*) FROM visits v JOIN patients p USING (patient_id)
              WHERE v.visit_date < p.registration_date),
           (SELECT COUNT(*) FROM visits),
           'visit_date precedes patient registration_date -> registration_date is not a reliable temporal anchor'

    UNION ALL
    SELECT 'billing_before_visit', 'WARN', 'billing + visits',
           (SELECT COUNT(*) FROM billing b JOIN visits v USING (visit_id)
              WHERE b.billing_date < v.visit_date),
           (SELECT COUNT(*) FROM billing),
           'billing_date precedes visit_date'

    -- Claim / amount integrity --------------------------------------------
    UNION ALL
    SELECT 'paid_missing_approved_amount', 'ERROR', 'billing',
           (SELECT COUNT(*) FROM billing WHERE claim_status = 'Paid' AND approved_amount IS NULL),
           (SELECT COUNT(*) FROM billing WHERE claim_status = 'Paid'),
           'Paid claims with no approved_amount'

    UNION ALL
    SELECT 'paid_missing_payment_days', 'WARN', 'billing',
           (SELECT COUNT(*) FROM billing WHERE claim_status = 'Paid' AND payment_days IS NULL),
           (SELECT COUNT(*) FROM billing WHERE claim_status = 'Paid'),
           'Paid claims with no payment_days'

    UNION ALL
    SELECT 'rejected_with_positive_approved', 'ERROR', 'billing',
           (SELECT COUNT(*) FROM billing WHERE claim_status = 'Rejected' AND COALESCE(approved_amount, 0) > 0),
           (SELECT COUNT(*) FROM billing WHERE claim_status = 'Rejected'),
           'Rejected claims that nonetheless carry an approved_amount'

    UNION ALL
    SELECT 'pending_with_approved_amount', 'INFO', 'billing',
           (SELECT COUNT(*) FROM billing WHERE claim_status = 'Pending' AND approved_amount IS NOT NULL),
           (SELECT COUNT(*) FROM billing WHERE claim_status = 'Pending'),
           'Pending claims that already carry an approved_amount'

    UNION ALL
    SELECT 'approved_exceeds_billed', 'ERROR', 'billing',
           (SELECT COUNT(*) FROM billing WHERE approved_amount > billed_amount),
           (SELECT COUNT(*) FROM billing),
           'approved_amount greater than billed_amount'

    -- Distribution artefacts --------------------------------------------
    UNION ALL
    SELECT 'los_at_floor_0_5h', 'WARN', 'visits',
           (SELECT COUNT(*) FROM visits WHERE length_of_stay_hours = 0.5),
           (SELECT COUNT(*) FROM visits),
           'Length of stay exactly 0.5h -> likely a clipped / censored minimum'

    UNION ALL
    SELECT 'billed_at_floor_500', 'WARN', 'billing',
           (SELECT COUNT(*) FROM billing WHERE billed_amount = 500.0),
           (SELECT COUNT(*) FROM billing),
           'billed_amount exactly 500 -> likely a floor value'

    UNION ALL
    SELECT 'approved_amount_missing_any', 'INFO', 'billing',
           (SELECT COUNT(*) FROM billing WHERE approved_amount IS NULL),
           (SELECT COUNT(*) FROM billing),
           'Billing rows with a NULL approved_amount (any status)'
)
SELECT
    check_name,
    severity,
    table_scope,
    records_flagged,
    records_total,
    ROUND(100.0 * records_flagged / NULLIF(records_total, 0), 2) AS pct_flagged,
    description
FROM checks
ORDER BY
    CASE severity WHEN 'ERROR' THEN 0 WHEN 'WARN' THEN 1 ELSE 2 END,
    records_flagged DESC;

COMMENT ON VIEW v_data_quality_report IS
    'Automated Phase 1 data-quality checks (severity ERROR/WARN/INFO) feeding Phase 2 EDA and Phase 6 governance';
