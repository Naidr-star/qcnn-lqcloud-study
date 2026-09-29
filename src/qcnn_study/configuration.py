"""Explicit compilation inputs, without historical device or route parameters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CompilationConfig:
    physical_layout: tuple[int, ...]
    initial_layout: tuple[int, ...]
    coupling_map: tuple[tuple[int, int], ...]
    basis_gates: tuple[str, ...]
    optimization_level: int
    transpiler_seed: int

    def __post_init__(self):
        layout, initial = self.physical_layout, self.initial_layout
        for values in (layout, initial):
            if not values or any(type(i) is not int or i < 0 for i in values) or len(set(values)) != len(values):
                raise ValueError("Layouts require distinct nonnegative integers")
        if not set(initial) <= set(layout):
            raise ValueError("Initial layout must belong to the selected physical layout")
        for edge in self.coupling_map:
            if len(edge) != 2 or edge[0] == edge[1] or any(type(i) is not int or i not in layout for i in edge):
                raise ValueError("Coupling edges must join two selected qubits")
        connected = {layout[0]}
        while True:
            larger = connected | {q for edge in self.coupling_map if connected.intersection(edge) for q in edge}
            if larger == connected:
                break
            connected = larger
        if connected != set(layout):
            raise ValueError("Selected physical subgraph must be connected")
        allowed = {"id", "rz", "sx", "sxdg", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "p", "u", "cx", "cz", "swap"}
        if not self.basis_gates or any(g not in allowed for g in self.basis_gates):
            raise ValueError("Unsupported compilation basis")
        if type(self.optimization_level) is not int or self.optimization_level not in range(4):
            raise ValueError("Invalid optimization level")
        if type(self.transpiler_seed) is not int or self.transpiler_seed < 0:
            raise ValueError("Invalid transpiler seed")
