from copy import deepcopy
from pathlib import Path
import json

import pytest

from qcnn_study.reproduction import analyze, load_inputs, render_tables
from qcnn_study.statistics import paired_exact

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs():
    return load_inputs(ROOT / "data")


def test_primary_outcomes_and_paired_counts(inputs):
    result = analyze(inputs)
    h = result["holdout"]
    assert [h[k]["correct"] for k in ("qpu_original_head", "qpu_calibrated", "ideal_calibrated")] == [100, 176, 180]
    assert h["qpu_original_head"]["prediction_class_counts"] == [200, 0]
    assert h["qpu_calibrated"]["confusion_matrix"] == [[83, 17], [7, 93]]
    pair = result["qpu_minus_ideal_paired"]
    assert (pair["candidate_only_correct"], pair["baseline_only_correct"]) == (11, 15)
    assert pair["exact_mcnemar_p"] == pytest.approx(0.557197093963623)


def test_temporal_identity_and_scores(inputs):
    result = analyze(inputs)
    assert [v["correct"] for v in result["temporal"].values()] == [72, 70, 70, 63, 67]
    assert result["refitted_minus_frozen_paired"]["session2"]["exact_mcnemar_p"] == pytest.approx(0.2890625)


def test_published_results_recalculate_exactly(inputs):
    result = analyze(inputs)
    assert json.loads((ROOT / "results/statistics.json").read_text(encoding="utf-8")) == result
    assert (ROOT / "results/tables.md").read_text(encoding="utf-8") == render_tables(result)


def test_row_order_does_not_change_results(inputs):
    reordered = deepcopy(inputs)
    reordered["holdout_predictions"]["records"].reverse()
    reordered["temporal_predictions"]["records"].reverse()
    assert analyze(reordered) == analyze(inputs)


@pytest.mark.parametrize("problem", ["duplicate", "missing", "invalid_class", "extra_field", "label_mismatch", "initial_mismatch", "unknown_id"])
def test_invalid_records_fail_closed(inputs, problem):
    h = inputs["holdout_predictions"]["records"]
    t = inputs["temporal_predictions"]["records"]
    if problem == "duplicate":
        h[1]["sample_id"] = h[0]["sample_id"]
    elif problem == "missing":
        h.pop()
    elif problem == "invalid_class":
        h[0]["qpu_calibrated"] = True
    elif problem == "extra_field":
        h[0]["unrecognized"] = 1
    elif problem == "label_mismatch":
        t[0]["label"] = 1-t[0]["label"]
    elif problem == "initial_mismatch":
        t[0]["initial"] = 1-t[0]["initial"]
    else:
        t[0]["sample_id"] = "H999"
    with pytest.raises(ValueError):
        analyze(inputs)


def test_integrity_binding_detects_input_change(tmp_path):
    source = ROOT / "data"
    for name in ("manifest.json", "benchmark_reference.json", "holdout_predictions.json", "study.json", "temporal_predictions.json"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    with (tmp_path / "study.json").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="integrity"):
        load_inputs(tmp_path)


def test_exact_paired_extremes():
    assert paired_exact([0, 1], [0, 1], [0, 1])["exact_mcnemar_p"] == 1
    assert paired_exact([0, 0], [1, 1], [1, 1])["exact_mcnemar_p"] == .5
    with pytest.raises(ValueError):
        paired_exact([], [], [])
