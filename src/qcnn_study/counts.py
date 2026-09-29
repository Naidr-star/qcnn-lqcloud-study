"""Integer raw counts, explicit bit order, and complete batch validation."""

import json

def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_counts(counts, width, shots):
    positive_int(width, "classical width")
    positive_int(shots, "shots")
    if not isinstance(counts, dict) or not counts:
        raise ValueError("Raw counts must be a nonempty mapping")
    for bits, count in counts.items():
        if not isinstance(bits, str) or len(bits) != width or set(bits)-{"0", "1"}:
            raise ValueError("Unexpected raw bitstring width or alphabet")
        if type(count) is not int or count < 0:
            raise ValueError("Raw counts must be nonnegative integers, not corrected weights")
    if sum(counts.values()) != shots:
        raise ValueError("Raw count total differs from requested shots")
    return dict(counts)


def to_qiskit_order(raw_counts, width, shots):
    """SDK classical-bit-zero-left -> Qiskit classical-bit-zero-right."""
    return {bits[::-1]: n for bits, n in validate_counts(raw_counts, width, shots).items()}


def z_expectation(raw_counts, width, shots, classical_bit=0):
    """An already basis-rotated observable, using SDK bit order."""
    counts = validate_counts(raw_counts, width, shots)
    if type(classical_bit) is not int or not 0 <= classical_bit < width:
        raise ValueError("Classical bit is outside the returned register")
    return sum((1 if bits[classical_bit] == "0" else -1)*n for bits, n in counts.items())/shots


def parse_completed(response, *, job_id, widths, shots, readout_correction):
    if response.get("task_id") != job_id or response.get("status") != "completed":
        raise ValueError("Response is not the requested completed task")
    payload = response.get("result")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("An inline batch response is required; remote result links are not followed")
    children = payload["results"]
    if len(children) != len(widths):
        raise ValueError("Incomplete or extra circuit results")
    records = []
    for ordinal, (child, width) in enumerate(zip(children, widths, strict=True)):
        if child.get("success") is not True or type(child.get("shots")) is not int or child.get("shots") != shots:
            raise ValueError("Unsuccessful result or changed shot count")
        declared = child.get("circuit_index")
        if declared is not None and (type(declared) is not int or declared != ordinal):
            raise ValueError("Server circuit index differs from request order")
        if child.get("dynamic_decoupling_applied"):
            raise ValueError("Unexpected dynamical decoupling")
        if readout_correction and (child.get("readout_correction_requested") is not True or child.get("readout_correction_applied") is not True):
            raise ValueError("Requested readout correction was not applied")
        counts = validate_counts(child.get("raw_counts"), width, shots)
        records.append({"ordinal": ordinal, "raw_counts": counts,
                        "qiskit_counts": to_qiskit_order(counts, width, shots),
                        "width": width, "shots": shots,
                        "order_verified_by_server_index": declared is not None})
    return records
