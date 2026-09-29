"""Offline semantic check using synthetic circuits and a fictitious topology."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qcnn_study.circuits import compile_circuit, native_unitary_roundtrip, terminal_measurements
from qcnn_study.configuration import CompilationConfig


def z_expectation(circuit, bit):
    p = Statevector.from_instruction(circuit).probabilities(qargs=[bit])
    return float(p[0] - p[1])


def main():
    # Synthetic values only; not historical device or calibration parameters.
    config = CompilationConfig((0, 1), (1, 0), ((0, 1),), ("rz", "sx", "x", "cx"), 1, 7)
    circuit = QuantumCircuit(2, 1)
    circuit.ry(0.37, 0)
    circuit.cx(0, 1)
    circuit.measure(1, 0)
    expected = z_expectation(circuit.remove_final_measurements(inplace=False), 1)
    compiled = compile_circuit(circuit, config)
    output_bit = terminal_measurements(compiled)[0]
    recovered = z_expectation(native_unitary_roundtrip(compiled), output_bit)
    error = abs(recovered - expected)
    if error > 1e-10:
        raise SystemExit("Synthetic circuit semantic check failed")
    print(f"Synthetic routed/native round-trip absolute error: {error:.3g}; cloud requests: 0")


if __name__ == "__main__":
    main()
