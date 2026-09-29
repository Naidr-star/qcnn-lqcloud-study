"""Explicit topology routing, output lightcones, and Qiskit/SDK translation."""

import math


def terminal_measurements(circuit):
    if circuit.parameters or not circuit.num_clbits:
        raise ValueError("Require bound circuits with explicit terminal measurements")
    seen, qubits, classical = False, set(), set()
    for item in circuit.data:
        gate = item.operation
        if getattr(gate, "condition", None) is not None or gate.name in {"if_else", "while_loop", "for_loop", "reset", "delay"}:
            raise ValueError("Dynamic control, reset and delay are outside the supported interchange subset")
        if gate.name == "measure":
            q = circuit.find_bit(item.qubits[0]).index
            c = circuit.find_bit(item.clbits[0]).index
            if q in qubits or c in classical:
                raise ValueError("Duplicate terminal measurement")
            qubits.add(q)
            classical.add(c)
            seen = True
        elif seen and gate.name != "barrier":
            raise ValueError("Only terminal measurements are supported")
    if classical != set(range(circuit.num_clbits)):
        raise ValueError("Every declared classical bit must be measured exactly once")
    return sorted(qubits)


def prune_to_outputs(source, outputs):
    """Unitary-source pruning only; it preserves the specified output observables."""
    from qiskit import QuantumCircuit
    outputs = tuple(outputs)
    if source.num_clbits or source.parameters or not outputs or any(type(q) is not int or not 0 <= q < source.num_qubits for q in outputs):
        raise ValueError("Supply valid outputs and a bound unmeasured unitary circuit")
    for item in source.data:
        if item.operation.name != "barrier":
            try:
                item.operation.to_matrix()
            except Exception:
                raise ValueError("Pruning requires explicit unitary gates") from None
    active, kept = set(outputs), []
    for item in reversed(source.data):
        bits = {source.find_bit(q).index for q in item.qubits}
        if item.operation.name != "barrier" and active.intersection(bits):
            kept.append(item)
            active.update(bits)
    target = QuantumCircuit(source.num_qubits)
    target.global_phase = source.global_phase
    for item in reversed(kept):
        target.append(item.operation, [source.find_bit(q).index for q in item.qubits])
    return target


def compile_circuit(source, config):
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    terminal_measurements(source)
    if len(config.initial_layout) != source.num_qubits:
        raise ValueError("Initial layout must cover each logical qubit")
    positions = {physical: i for i, physical in enumerate(config.physical_layout)}
    coupling = CouplingMap()
    for bit in range(len(positions)):
        coupling.add_physical_qubit(bit)
    for a, b in config.coupling_map:
        coupling.add_edge(positions[a], positions[b])
        coupling.add_edge(positions[b], positions[a])
    compiled = transpile(source, basis_gates=list(config.basis_gates), coupling_map=coupling,
                         initial_layout=[positions[q] for q in config.initial_layout],
                         optimization_level=config.optimization_level, seed_transpiler=config.transpiler_seed)
    if compiled.num_qubits != len(config.physical_layout):
        raise ValueError("Routed circuit width differs from explicit physical layout")
    terminal_measurements(compiled)
    return compiled


def to_lqcloud(source):
    from lqcloud import QuantumCircuit

    terminal_measurements(source)
    target = QuantumCircuit(source.num_qubits, source.num_clbits)
    measuring = False
    for item in source.data:
        gate = item.operation
        name = gate.name
        bits = [source.find_bit(q).index for q in item.qubits]
        cbits = [source.find_bit(c).index for c in item.clbits]
        if name == "measure":
            if not measuring:
                target.barrier()
                measuring = True
            target.measure(bits[0], cbits[0])
        elif name == "barrier":
            target.barrier(*bits)
        elif name in {"id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg"}:
            getattr(target, name)(bits[0])
        elif name in {"rx", "ry", "rz", "p"}:
            angle = float(gate.params[0])
            if not math.isfinite(angle):
                raise ValueError("Nonfinite rotation")
            getattr(target, name)(angle, bits[0])
        elif name in {"cx", "cz", "swap"}:
            getattr(target, name)(*bits)
        elif name in {"u", "unitary"}:
            import numpy as np
            matrix = gate.to_matrix()
            if not np.isfinite(matrix).all():
                raise ValueError("Nonfinite unitary")
            target.unitary(matrix, bits)
        else:
            raise ValueError("Unsupported gate; compile to the documented basis first")
    return target


def native_unitary_roundtrip(source):
    """Two-level semantic check only; it does not simulate physical readout pulses."""
    from qiskit import QuantumCircuit

    native = to_lqcloud(source).decompose()
    restored = QuantumCircuit(len(native.qubits))
    for gate, qubits, _ in native.data:
        bits = [native.qubits.index(q) for q in qubits]
        if gate.name in {"barrier", "measure"}:
            continue
        if gate.name == "su2":
            restored.unitary(gate.matrix(), bits)
        elif gate.name in {"id", "x", "y", "z", "cz"}:
            getattr(restored, gate.name)(*bits)
        elif gate.name == "rz":
            restored.rz(float(gate.params[0]), bits[0])
        elif gate.name in {"xhalf", "mxhalf", "yhalf", "myhalf"}:
            angle = (-1 if gate.name.startswith("m") else 1)*math.pi/2
            (restored.rx if "x" in gate.name else restored.ry)(angle, bits[0])
        else:
            raise ValueError("Unknown native gate in local semantic checker")
    return restored
