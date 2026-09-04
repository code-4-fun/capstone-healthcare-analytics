"""The request validation gate (reuses the Phase 2 rule registry)."""
from __future__ import annotations

import pandas as pd

from capstone import monitoring as mon

from conftest import sample_reference_rows


def _payloads(n=80):
    rows = sample_reference_rows("B", n=n)
    return [mon.feature_row_to_payload(r, "B") for _, r in rows.iterrows()]


def _frame(payloads):
    return pd.json_normalize(payloads).assign(request_id=[str(i) for i in range(len(payloads))])


def test_clean_batch_passes():
    res = mon.validation_gate(_frame(_payloads()))
    assert res.fail_rate == 0.0
    assert res.passed
    assert (res.offences["n_offending"] == 0).all()


def test_bad_enum_negative_amount_and_out_of_range_age_are_caught():
    payloads = _payloads(60)
    payloads[0]["department"] = "Radiology"          # not a known department
    payloads[1]["gender"] = "X"
    payloads[2]["age"] = 250
    payloads[3]["billed_amount"] = -10.0
    payloads[4]["length_of_stay_hours"] = -1.0

    res = mon.validation_gate(_frame(payloads))
    assert not res.passed
    assert len(res.offending_ids) == 5

    by_rule = res.offences.set_index("rule")["n_offending"]
    assert by_rule["department_enum"] == 1
    assert by_rule["gender_enum"] == 1
    assert by_rule["age_range"] == 1
    assert by_rule["billed_negative"] == 1
    assert by_rule["los_negative"] == 1


def test_gate_rules_are_a_subset_of_the_phase2_registry():
    from capstone import data_quality as dq

    for name in mon.GATE_RULES:
        assert name in dq.RULES_BY_NAME
