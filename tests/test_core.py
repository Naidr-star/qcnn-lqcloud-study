import numpy as np
import pytest

from qcnn_study.counts import parse_completed, to_qiskit_order, validate_counts, z_expectation
from qcnn_study.models import CentroidModel, evaluate_holdout, majority_predict
from qcnn_study.statistics import accuracy_report, paired_comparison, wilson_interval


def test_counts_bit_order_and_expectations():
    counts = {"01": 9, "10": 4}
    assert to_qiskit_order(counts, 2, 13) == {"10": 9, "01": 4}
    assert z_expectation(counts, 2, 13, 0) == 5/13
    assert z_expectation(counts, 2, 13, 1) == -5/13


@pytest.mark.parametrize("counts", [{"0": 1.0}, {"0": True}, {"0": -1, "1": 14}, {"00": 13}, {"x": 13}, {"0": 12}, {}])
def test_raw_counts_reject_nonraw_or_invalid(counts):
    with pytest.raises(ValueError):
        validate_counts(counts, 1, 13)


def test_centroid_and_constant_feature():
    model = CentroidModel.fit([[-2, 3], [-1, 3], [1, 3], [2, 3]], [0, 0, 1, 1])
    assert model.scale[1] == 1
    assert model.predict([[-.5, 3], [.5, 3]]).tolist() == [0, 1]
    assert model.predict([[0, 3]]).tolist() == [0]
    assert majority_predict([model]*3, [[[-.5, 3], [.5, 3]]]*3).tolist() == [0, 1]
    with pytest.raises(ValueError):
        majority_predict([model]*2, [[[-.5, 3]]]*2)


def test_holdout_separation_and_no_test_label_fitting():
    kwargs = dict(training=[[[-1], [1]]], training_labels=[0, 1], holdout=[[[-.5], [.5]]],
                  holdout_labels=[0, 1], training_ids=["a", "b"], holdout_ids=["c", "d"])
    assert evaluate_holdout(**kwargs)["correct"] == 2
    assert evaluate_holdout(**{**kwargs, "holdout_labels": [1, 0]})["correct"] == 0
    with pytest.raises(ValueError, match="overlap"):
        evaluate_holdout(**{**kwargs, "holdout_ids": ["a", "d"]})
    with pytest.raises(ValueError):
        CentroidModel.fit([[1], [2]], [0, 0])


def test_wilson_and_confusion():
    # Generic binomial regression case, unrelated to any cloud experiment.
    interval = wilson_interval(81, 263)
    assert np.round(np.asarray(interval), 4).tolist() == [.2553, .3662]
    assert accuracy_report([0, 0, 1, 1], [0, 1, 0, 1])["confusion_matrix"] == [[1, 1], [1, 1]]
    assert wilson_interval(0, 8)[0] == 0
    with pytest.raises(ValueError):
        wilson_interval(2, 1)


def test_paired_statistics_are_paired_and_reproducible():
    result = paired_comparison([0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], draws=101, seed=4)
    assert result["candidate_only_correct"] == 2
    assert result["baseline_only_correct"] == 0
    assert result["exact_mcnemar_p"] == .5
    assert result == paired_comparison([0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], draws=101, seed=4)


def test_completed_response_identity_counts_and_order():
    child = dict(success=True, shots=13, raw_counts={"0": 8, "1": 5})
    response = dict(task_id="mock-job", status="completed", result={"results": [child]})
    kwargs = dict(job_id="mock-job", widths=[1], shots=13, readout_correction=False)
    assert parse_completed(response, **kwargs)[0]["order_verified_by_server_index"] is False
    child["circuit_index"] = 1
    with pytest.raises(ValueError):
        parse_completed(response, **kwargs)
    child["circuit_index"] = 0
    with pytest.raises(ValueError):
        parse_completed(response, **{**kwargs, "readout_correction": True})
