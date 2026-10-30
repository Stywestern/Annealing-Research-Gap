"""Step 1: hardware description (Zephyr graph + programmable ranges) and annealing schedule.

Two ways to get a Hardware object:
  * Hardware.ideal_zephyr(m=12, t=4)  -> perfect Zephyr Z(12,4), no defects, offline.
  * Hardware.from_qpu()               -> ONE API call (no QPU time used) to fetch the real
                                         working graph + ranges; save it with .save() and
                                         afterwards work fully offline with Hardware.load().
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import os

KB_OVER_H_GHZ_PER_K = 20.8366   # k_B / h expressed in GHz per Kelvin

@dataclass
class Hardware:
    graph: nx.Graph
    name: str = "ideal_zephyr"
    topology: dict = field(default_factory=lambda: {"type": "zephyr", "shape": [15, 4]})
    # Programmable ranges. DEFAULTS ARE ASSUMPTIONS for an Advantage2-like device;
    # from_qpu() overwrites them with the values the real solver reports.
    h_range: tuple = (-4.0, 4.0)
    j_range: tuple = (-1.0, 1.0)
    extended_j_range: tuple = (-2.0, 1.0)
    temperature_mk: float = 12.0          # physical fridge temperature; treated as a fit parameter later
    extra: dict = field(default_factory=dict)

    # ---- constructors ------------------------------------------------------
    @classmethod
    def ideal_zephyr(cls, m: int = 15, t: int = 4) -> "Hardware":
        import dwave_networkx as dnx
        return cls(graph=dnx.zephyr_graph(m, t), name=f"ideal_zephyr_Z({m},{t})",
                   topology={"type": "zephyr", "shape": [m, t]})

    @classmethod
    def from_qpu(cls, solver=None) -> "Hardware":
        """Needs a Leap API token. Reads solver properties only (uses no QPU time)."""
        from dwave.system import DWaveSampler
        sampler = DWaveSampler(solver=solver or dict(topology__type="zephyr"))
        p = sampler.properties
        return cls(
            graph=sampler.to_networkx_graph(),
            name=sampler.solver.name,
            topology=dict(p["topology"]),
            h_range=tuple(p["h_range"]),
            j_range=tuple(p["j_range"]),
            extended_j_range=tuple(p.get("extended_j_range", p["j_range"])),
            extra={"chip_id": p.get("chip_id"),
                   "annealing_time_range": p.get("annealing_time_range"),
                   "default_annealing_time": p.get("default_annealing_time")},
        )

    # ---- persistence ---------------------------------------------------------
    def save(self, path) -> None:
        payload = {"name": self.name, "topology": self.topology,
                   "h_range": self.h_range, "j_range": self.j_range,
                   "extended_j_range": self.extended_j_range,
                   "temperature_mk": self.temperature_mk, "extra": self.extra,
                   "nodes": sorted(self.graph.nodes), "edges": sorted(self.graph.edges)}
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path) -> "Hardware":
        d = json.loads(Path(path).read_text())
        g = nx.Graph()
        g.add_nodes_from(d["nodes"])
        g.add_edges_from(map(tuple, d["edges"]))
        return cls(graph=g, name=d["name"], topology=d["topology"],
                   h_range=tuple(d["h_range"]), j_range=tuple(d["j_range"]),
                   extended_j_range=tuple(d["extended_j_range"]),
                   temperature_mk=d["temperature_mk"], extra=d.get("extra", {}))

    def summary(self) -> str:
        n, e = self.graph.number_of_nodes(), self.graph.number_of_edges()
        return f"{self.name}: {n} qubits, {e} couplers, topology={self.topology}"


@dataclass
class Schedule:
    """Annealing schedule: A(s) transverse, B(s) problem energy scale, both in GHz."""
    s: np.ndarray
    A: np.ndarray
    B: np.ndarray
    is_placeholder: bool = False

    @classmethod
    def from_csv(cls, path: str = "/home/stywestern/Quantum_Digital_Twin/solvers/simulated_quantum_annealing/standart_annealing_schedule_Ad2Sys1.csv") -> "Schedule":
        """CSV from D-Wave's docs (QPU-specific anneal schedule): columns s, A(s), B(s), ...
        One header row is skipped. Falls back to placeholder if the file is not found."""
        if os.path.exists(path):
            try:
                arr = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1, 2))
                return cls(s=arr[:, 0], A=arr[:, 1], B=arr[:, 2])
            except Exception as e:
                warnings.warn(f"Error reading {path}: {e}. Falling back to placeholder schedule.")
        else:
            warnings.warn(f"Schedule file not found at '{path}'. Using PLACEHOLDER annealing schedule.")
        
        return cls.placeholder()

    @classmethod
    def placeholder(cls) -> "Schedule":
        """Smooth made-up curves with the right qualitative shape (A falls, B rises, crossing near
        s~0.4-0.5). NOT the real device schedule: replace with from_csv() before drawing conclusions."""
        warnings.warn("Using PLACEHOLDER annealing schedule; load the real one with Schedule.from_csv().")
        s = np.linspace(0.0, 1.0, 501)
        return cls(s=s, A=8.0 * (1.0 - s) ** 2.5, B=12.0 * s ** 1.5, is_placeholder=True)

    def at(self, s: float):
        return float(np.interp(s, self.s, self.A)), float(np.interp(s, self.s, self.B))


def beta_eff(B_ghz: float, temperature_mk: float) -> float:
    """Inverse temperature per unit of *scaled* Ising energy at freeze-out point with scale B (GHz):
    weight = exp(-(B/2) E_scaled / kT)  ->  beta = (B/2) / kT (kT in GHz)."""
    kT = KB_OVER_H_GHZ_PER_K * temperature_mk * 1e-3
    return 0.5 * B_ghz / kT

# =========================================================================
# Execution Block: Self-Test
# =========================================================================
if __name__ == "__main__":
    import os

    print("=== Testing Hardware Offline Initialization ===")
    # 1. Create a smaller ideal Zephyr graph for a quick test
    hw_ideal = Hardware.ideal_zephyr(m=4, t=4)
    print(f"[+] Initialized: {hw_ideal.summary()}")
    print(f"[+] Default Ranges: h={hw_ideal.h_range}, j={hw_ideal.j_range}")

    print("\n=== Testing Persistence (Save/Load) ===")
    test_filepath = "test_hardware_snapshot.json"
    
    # 2. Save it to disk
    hw_ideal.save(test_filepath)
    print(f"[+] Saved snapshot to {test_filepath}")
    
    # 3. Load it back
    hw_loaded = Hardware.load(test_filepath)
    print(f"[+] Loaded snapshot: {hw_loaded.summary()}")
    
    # Clean up the test file
    #if os.path.exists(test_filepath):
    #    os.remove(test_filepath)
    #    print(f"[+] Cleaned up {test_filepath}")

    print("\n=== Testing Annealing Schedule & Interpolation ===")
    # 4. Generate mock CSV, load both schedules, and compare
    csv_path = "solvers/simulated_quantum_annealing/standart_annealing_schedule_Ad2Sys1.csv"

    schedule_ph = Schedule.placeholder()
    schedule_csv = Schedule.from_csv(csv_path)
    
    s_val = 0.5
    a_ph, b_ph = schedule_ph.at(s_val)
    a_csv, b_csv = schedule_csv.at(s_val)
    
    print(f"[+] At anneal fraction s={s_val}:")
    print(f"    Placeholder   : A(s) = {a_ph:.3f} GHz, B(s) = {b_ph:.3f} GHz")
    print(f"    CSV           : A(s) = {a_csv:.3f} GHz, B(s) = {b_csv:.3f} GHz")
    print(f"    Difference    : \u0394A = {abs(a_ph - a_csv):.3f} GHz, \u0394B = {abs(b_ph - b_csv):.3f} GHz")

    print("\n=== Testing Thermodynamic Physics ===")
    # 5. Test the effective beta calculation using the placeholder B value
    temp_mk = 12.0  # 12 millikelvin fridge
    beta = beta_eff(B_ghz=b_ph, temperature_mk=temp_mk)
    print(f"[+] Effective beta at {temp_mk} mK and B(s)={b_ph:.3f} GHz : {beta:.4f}")

    print("\n=== Testing Live QPU Fetch ===")
    # 6. Try fetching from D-Wave Leap (Expected to fail without token)
    try:
        hw_live = Hardware.from_qpu()
        print("[+] SUCCESS! Fetched live hardware data:")
        print(f"    {hw_live.summary()}")
        print(f"    Real h_range: {hw_live.h_range}")
        print(f"    Real j_range: {hw_live.j_range}")
    except Exception as e:
        print(f"[-] Bypassed live fetch (No Leap Token or Connection): {e}")
        
    print("\n[+] All offline hardware methods executed successfully!")