import pytest

pytest.importorskip("qiskit", reason="Install the quantum extra for circuit tests")
pytest.importorskip("lqcloud", reason="Install the quantum extra for SDK round-trip tests")

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qcnn_study.circuits import compile_circuit, native_unitary_roundtrip, prune_to_outputs, terminal_measurements, to_lqcloud
from qcnn_study.configuration import CompilationConfig


def expectation(circuit, bit):
    p = Statevector.from_instruction(circuit).probabilities(qargs=[bit])
    return float(p[0]-p[1])


def test_nontrivial_layout_preserves_measured_output():
    config = CompilationConfig((5, 9), (9, 5), ((5, 9),), ("rz", "sx", "x", "cx"), 1, 7)
    c = QuantumCircuit(2, 1)
    c.ry(.37, 0)
    c.cx(0, 1)
    c.measure(1, 0)
    compiled = compile_circuit(c, config)
    output = terminal_measurements(compiled)[0]
    assert expectation(c.remove_final_measurements(inplace=False), 1) == pytest.approx(expectation(native_unitary_roundtrip(compiled), output), abs=1e-10)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_pauli_signs(axis, sign):
    c = QuantumCircuit(1, 1)
    if sign < 0:
        c.x(0)
    if axis in ("x", "y"):
        c.h(0)
    if axis == "y":
        c.s(0)
        c.sdg(0)
    if axis in ("x", "y"):
        c.h(0)
    c.measure(0, 0)
    config = CompilationConfig((0,), (0,), (), ("rz", "sx", "x"), 0, 7)
    compiled = compile_circuit(c, config)
    assert expectation(native_unitary_roundtrip(compiled), 0) == pytest.approx(sign, abs=1e-10)


@pytest.mark.parametrize("angle", [.17, -.29, 1.1])
def test_noncommuting_rotation_roundtrip(angle):
    c = QuantumCircuit(1, 1)
    c.ry(angle, 0)
    c.rz(-.2, 0)
    c.sx(0)
    c.measure(0, 0)
    assert expectation(c.remove_final_measurements(inplace=False), 0) == pytest.approx(expectation(native_unitary_roundtrip(c), 0), abs=1e-10)


def test_output_lightcone_preserves_target():
    c = QuantumCircuit(3)
    c.ry(.7, 0)
    c.cx(0, 1)
    c.rx(.3, 2)
    pruned = prune_to_outputs(c, [1])
    assert len(pruned.data) < len(c.data)
    assert expectation(c, 1) == pytest.approx(expectation(pruned, 1), abs=1e-12)


def test_invalid_measurement_structure_rejected():
    c = QuantumCircuit(1, 1)
    c.measure(0, 0)
    c.x(0)
    with pytest.raises(ValueError):
        to_lqcloud(c)
    c = QuantumCircuit(1, 2)
    c.measure(0, 0)
    with pytest.raises(ValueError):
        terminal_measurements(c)


def test_unlowered_gates_are_not_certified_as_native():
    c = QuantumCircuit(1, 1)
    c.h(0)
    c.measure(0, 0)
    with pytest.raises(ValueError, match="Unknown native gate"):
        native_unitary_roundtrip(c)


@pytest.mark.parametrize("layout,initial,edges", [((0, 0), (0,), ()), ((0, 1), (2,), ((0, 1),)), ((0, 1), (0, 1), ())])
def test_invalid_topology_rejected(layout, initial, edges):
    with pytest.raises(ValueError):
        CompilationConfig(layout, initial, edges, ("rz", "sx", "x", "cx"), 1, 7)
