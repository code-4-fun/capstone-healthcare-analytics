-- Phase 1 :: Reusable business queries
-- Leadership-facing questions answered directly off the analytics views.
-- Run ad hoc:  psql ... -f sql/06_business_queries.sql   (or copy individually)

SET search_path TO capstone_solution, public;

-- Q1. Which departments carry the most operational risk (high-risk load + LOS)?
SELECT department, visits, high_risk_visits, high_risk_pct, avg_los_hours, icu_pct
FROM v_department_performance
ORDER BY high_risk_pct DESC;

-- Q2. Where is revenue leaking the most (denials vs pending), by department?
SELECT department, total_billed, total_collected, realization_rate_pct,
       total_leakage, total_pending, rejection_rate_pct
FROM v_department_performance
ORDER BY total_leakage DESC;

-- Q3. Which insurer is slowest to pay and most likely to reject?
SELECT insurance_provider, claims, rejection_rate_pct, pending_rate_pct,
       avg_approved_ratio_pct, avg_payment_days, p90_payment_days, realization_rate_pct
FROM v_insurance_provider_behavior
ORDER BY rejection_rate_pct DESC, avg_payment_days DESC;

-- Q4. Monthly revenue realization trend (is collection improving or worsening?)
SELECT billing_month, billed_amount, collected_amount, realization_rate_pct,
       pending_amount, leakage_rate_pct
FROM v_revenue_realization_monthly
ORDER BY billing_month;

-- Q5. What claim characteristics concentrate rejections?
SELECT dimension, value, claims, rejected, rejection_rate_pct
FROM v_claim_rejection_analysis
ORDER BY rejection_rate_pct DESC
LIMIT 15;

-- Q6. Patient-flow seasonality: busiest and quietest weeks
SELECT date_trunc('week', visit_date)::date AS week,
       SUM(visits) AS visits, SUM(high_risk_visits) AS high_risk_visits,
       ROUND(AVG(avg_los_hours), 2) AS avg_los_hours
FROM v_patient_flow_daily
GROUP BY 1
ORDER BY visits DESC;

-- Q7. Top 20 doctors by volume and their claim rejection rate
SELECT doctor_id, visits, unique_patients, departments_covered,
       high_risk_visits, avg_los_hours, rejection_rate_pct
FROM v_doctor_workload
ORDER BY visits DESC
LIMIT 20;

-- Q8. High-value claims still pending (cash-flow watchlist)
SELECT visit_id, department, insurance_provider, billed_amount, billing_date
FROM v_visit_billing
WHERE claim_status = 'Pending' AND billed_amount >= 30000
ORDER BY billed_amount DESC
LIMIT 25;

-- Q9. Chronic vs non-chronic: utilisation and acuity
SELECT chronic_flag,
       COUNT(*) AS visits,
       COUNT(DISTINCT patient_id) AS patients,
       ROUND(COUNT(*)::numeric / COUNT(DISTINCT patient_id), 2) AS visits_per_patient,
       ROUND(100.0 * COUNT(*) FILTER (WHERE risk_score = 'High') / COUNT(*), 1) AS high_risk_pct,
       ROUND(AVG(length_of_stay_hours), 2) AS avg_los_hours
FROM v_visit_billing
GROUP BY chronic_flag;

-- Q10. City-level revenue and rejection profile
SELECT city,
       COUNT(*) AS visits,
       SUM(billed_amount) AS total_billed,
       ROUND(100.0 * SUM(collected_amount) / NULLIF(SUM(billed_amount), 0), 1) AS realization_rate_pct,
       ROUND(100.0 * COUNT(*) FILTER (WHERE is_rejected) / COUNT(*), 1) AS rejection_rate_pct
FROM v_visit_billing
GROUP BY city
ORDER BY total_billed DESC;
