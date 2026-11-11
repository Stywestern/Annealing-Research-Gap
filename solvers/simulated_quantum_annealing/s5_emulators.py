"""Steps 2-4 glued together: EmulatedQPU.sample(logical_bqm, embedding, ...) -> EmulationResult.

Pipeline per call:
  chain strength -> embed -> scale to the device ranges -> quantise (DAC) -> [per programming cycle: optional
  spin-reversal gauge, control-error noise, sampler] -> raw chain resolution -> optional physical descent ->
  chain resolution -> optional logical descent.

Everything that defines or happened in a run is recorded:
  EmulatedQPU.provenance()   first collects the provenance of the submodules (hardware, schedule, sampler,
                             postprocessing engines), then adds the emulator's own settings, seed, temperature
                             source, software versions and a config hash.
  EmulationResult.info       what THIS run did: timing, seeds per programming cycle, mapping diagnostics, thermal
                             resolution, embedding footprint, chain breaks BEFORE any descent, per-stage energies,
                             postprocessing reports, and (by default) a provenance snapshot.
Postprocessing engines are per emulator (not the module-level ones in s4_postprocess).
"""
from __future__ import annotations

import contextlib
import copy
import datetime as _dt
import hashlib
import inspect
import json
import platform
import random as _pyrandom
import time
from dataclasses import dataclass, field
from importlib import metadata as _md

import dimod
import numpy as np
from dwave.embedding import (MinimizeEnergy, discard, embed_bqm, majority_vote,
                             unembed_sampleset, weighted_random)
from dwave.embedding.chain_strength import uniform_torque_compensation

from solvers.simulated_quantum_annealing.s1_hardware import Hardware, Schedule
from solvers.simulated_quantum_annealing.s1b_embedding import embedding_footprint, embedding_stats
from solvers.simulated_quantum_annealing.s2_mapping import (IsingArrays, SystematicICE, add_ice_noise, apply_gauge,
                                                            mapping_diagnostics, quantise, random_gauge,
                                                            scale_to_hardware, undo_gauge)
from solvers.simulated_quantum_annealing.s3_samplers import make_sampler
from solvers.simulated_quantum_annealing.s4_postprocess import GreedyDescent, get_config as pp_get_config

_POSTPROCESS = (None, "physical", "logical", "both")
_CB_METHODS = ("majority", "discard", "weighted", "minimize_energy")
_CYCLE_KEYS = ("beta_used", "beta_nominal", "freeze_s_used", "mean_energy", "min_energy")
_PACKAGES = ("numpy", "scipy", "networkx", "dimod", "dwave-system", "dwave-samplers", "dwave-networkx",
             "minorminer", "pandapower")


# ----------------------------------------------------------------------------------------------- helpers
def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    return repr(o)


@contextlib.contextmanager
def _seeded_global_rngs(seed: int):
    """Chain-break resolution in Ocean may draw from the global numpy / python generators (tie-breaking, weighted
    random). Seeding them for the duration of the call makes a run reproducible whichever one it uses; the previous
    global state is restored afterwards."""
    np_state, py_state = np.random.get_state(), _pyrandom.getstate()
    np.random.seed(int(seed) % (2 ** 32))
    _pyrandom.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(np_state)
        _pyrandom.setstate(py_state)


def _software_versions() -> dict:
    out = {"python": platform.python_version()}
    for p in _PACKAGES:
        try:
            out[p] = _md.version(p)
        except _md.PackageNotFoundError:
            pass
    return out


def _mean(a):
    a = np.asarray(a, dtype=float)
    return float(a.mean()) if a.size else None


@dataclass
class EmulationResult:
    sampleset: dimod.SampleSet            # logical, BINARY, has 'chain_break_fraction'; the stage chosen by `postprocess`
    chain_strength: float                 # in problem units, before hardware scaling
    scale_factor: float                   # problem units -> device units
    embedding_stats: dict
    target_sampleset: dimod.SampleSet | None = None   # physical samples of the final physical stage (keep_target=True)
    info: dict = field(default_factory=dict)
    raw_sampleset: dimod.SampleSet | None = None      # logical, BINARY: chains resolved, NO descent of any kind
    stages: dict | None = None                        # keep_stages=True: {"raw", "physical", "logical", "both"}


class EmulatedQPU:
    def __init__(self, hardware: Hardware, sampler: str = "sa",
                 sampler_kwargs: dict | None = None, dac_bits: int | None = 5,
                 ice_sigma_h: float = 0.02, ice_sigma_j: float = 0.01,
                 use_extended_j: bool = True, seed: int = 0, *,
                 schedule: Schedule | None = None, schedule_path: str | None = None, strict_schedule: bool = True,
                 temperature_mk: float | None = None,
                 dac_bits_h: int | None = None, dac_bits_j: int | None = None,
                 ice_sigma_j_chain: float | None = None, systematic: SystematicICE | None = None,
                 per_qubit_coupling_range=None, postprocess_settings: dict | None = None):
        """
        dac_bits, ice_sigma_h, ice_sigma_j : PLACEHOLDER defaults, calibration knobs; no physical meaning yet.
        use_extended_j : allow chain couplers (strongly negative J) to use the extended J range (an assumption about
                         the hardware that still has to be checked against D-Wave's documentation).
        schedule / schedule_path : a Schedule object, or a CSV path (default: the QPU_SCHEDULE_CSV environment variable
                         or the module default). strict_schedule=True raises if the file is missing, so a result run can
                         never silently use a placeholder.
        temperature_mk : overrides the sampler temperature. Otherwise hardware.effective_temperature_mk is used when
                         it has been fitted, else the physical hardware.temperature_mk. The source is recorded.
        dac_bits_h / dac_bits_j : separate DAC resolutions for h and J (default: dac_bits for both).
        ice_sigma_j_chain : separate control-error sigma for chain couplers (default: ice_sigma_j for all couplers).
        systematic : a SystematicICE (fixed per-qubit / per-coupler offsets that belong to the device).
        per_qubit_coupling_range : None, (lo, hi), or "from_hardware" (taken from hardware.properties, which exist only
                         for a device loaded with Hardware.from_qpu()). Limits the signed sum of couplings per qubit.
        postprocess_settings : dict of GreedyDescent settings for this emulator's own descent engines (for example
                         {"two_flip": True}); the defaults come from s4_postprocess.get_config().
        """
        self.hw = hardware
        if schedule is not None:
            self.schedule = schedule
        else:
            self.schedule = Schedule.from_csv(schedule_path, strict=strict_schedule)

        if temperature_mk is not None:
            self.temperature_used_mk, self.temperature_source = float(temperature_mk), "override"
        elif getattr(hardware, "effective_temperature_mk", None) is not None:
            self.temperature_used_mk, self.temperature_source = float(hardware.effective_temperature_mk), "effective"
        else:
            self.temperature_used_mk, self.temperature_source = float(hardware.temperature_mk), "physical"

        self.sampler_name = sampler
        self.sampler = make_sampler(sampler, self.schedule, self.temperature_used_mk, **(sampler_kwargs or {}))
        try:
            self._sampler_takes_anneal_time = "anneal_time_us" in inspect.signature(self.sampler.sample).parameters
        except (TypeError, ValueError):
            self._sampler_takes_anneal_time = False

        self.dac_bits, self.dac_bits_h, self.dac_bits_j = dac_bits, dac_bits_h, dac_bits_j
        self.sigma_h, self.sigma_j, self.sigma_j_chain = ice_sigma_h, ice_sigma_j, ice_sigma_j_chain
        self.use_extended_j = use_extended_j
        if systematic is not None and not hasattr(systematic, "offsets"):
            raise TypeError("systematic must be a SystematicICE (an object with an .offsets method)")
        self.systematic = systematic

        if per_qubit_coupling_range == "from_hardware":
            prop = (getattr(hardware, "properties", None) or {}).get("per_qubit_coupling_range")
            if not prop:
                raise ValueError("this Hardware has no 'per_qubit_coupling_range' property (it exists only for a device "
                                 "loaded with Hardware.from_qpu()); pass an explicit (lo, hi) or None")
            per_qubit_coupling_range = prop
        self.per_qubit_range = None if per_qubit_coupling_range is None else (
            float(per_qubit_coupling_range[0]), float(per_qubit_coupling_range[1]))

        cfg = {**pp_get_config(), **(postprocess_settings or {})}
        self.pp_physical = GreedyDescent(label="physical", **cfg)     # unknown settings raise TypeError here
        self.pp_logical = GreedyDescent(label="logical", **cfg)

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._created = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        self._runs = 0

    # ------------------------------------------------------------------ small internals
    def _chain_j_range(self):
        return self.hw.extended_j_range if self.use_extended_j else self.hw.j_range

    @staticmethod
    def _chain_break_method(name, logical_spin, embedding):
        if name == "majority":
            return majority_vote
        if name == "discard":
            return discard
        if name == "weighted":
            return weighted_random
        if name == "minimize_energy":
            return MinimizeEnergy(logical_spin, embedding)
        raise ValueError(f"chain_break_method must be one of {_CB_METHODS}, got {name!r}")

    @staticmethod
    def _local_chain_cap(logical_spin: dimod.BinaryQuadraticModel, embedding: dict,
                         relative_chain_multiplier: float) -> dict:
        """Per-variable cap = relative_chain_multiplier * (sum of |coupling| on edges actually
        incident to that logical variable). A chain only needs to outlast the pull it is actually
        under, so this is anchored locally rather than to the single worst coefficient anywhere in
        the whole problem (which earlier under- or over-constrained every OTHER chain at once)."""
        incident = {v: 0.0 for v in logical_spin.variables}
        for u, v, bias in logical_spin.iter_quadratic():
            incident[u] += abs(bias)
            incident[v] += abs(bias)
        return {v: relative_chain_multiplier * incident[v] for v in embedding}

    @staticmethod
    def _target(samples, ideal, embedded):
        return dimod.SampleSet.from_samples((samples, ideal.labels), energy=embedded.energies((samples, ideal.labels)),
                                            vartype=dimod.SPIN)

    def _resolve_chains(self, target, embedding, logical_spin, method_name):
        method = self._chain_break_method(method_name, logical_spin, embedding)
        seed = int(self.rng.integers(2 ** 31 - 1))
        with _seeded_global_rngs(seed):
            return unembed_sampleset(target, embedding, logical_spin, chain_break_method=method,
                                     chain_break_fraction=True)

    def _descend_logical(self, ss: dimod.SampleSet, spin_bqm: dimod.BinaryQuadraticModel):
        """Logical-level descent with THIS emulator's engine. Same bookkeeping as s4_postprocess.descend_sampleset
        (kept separate on purpose so that s4 stays untouched): keeps num_occurrences, extra vectors and info, adds the
        call report as info['postprocess_descent'], recomputes energies. Returns (sampleset, flips, call_info)."""
        ar = IsingArrays.from_bqm(spin_bqm)
        pos = {v: i for i, v in enumerate(ss.variables)}
        arr = np.asarray(ss.record.sample)[:, [pos[v] for v in ar.labels]]
        new, flips, call_info = self.pp_logical.run(ar, arr)
        energies = spin_bqm.energies((new, ar.labels))
        extra = {k: ss.record[k] for k in ss.record.dtype.names if k not in ("sample", "energy", "num_occurrences")}
        meta = dict(ss.info)
        meta["postprocess_descent"] = copy.deepcopy(call_info)
        out = dimod.SampleSet.from_samples((new, ar.labels), energy=energies, vartype=dimod.SPIN,
                                           num_occurrences=ss.record.num_occurrences, info=meta, **extra)
        return out, int(flips), call_info

    @staticmethod
    def _stage_summary(ss: dimod.SampleSet) -> dict:
        e = np.asarray(ss.record.energy, dtype=float)
        return {"n_rows": int(len(ss)), "mean_energy": _mean(e), "min_energy": float(e.min()) if e.size else None}

    # ------------------------------------------------------------------ provenance
    def _sampler_description(self) -> dict:
        if hasattr(self.sampler, "describe"):
            d = dict(self.sampler.describe())
        else:
            d = {k: v for k, v in vars(self.sampler).items() if not k.startswith("_") and k != "schedule"}
        d.setdefault("class", type(self.sampler).__name__)
        return _json_safe(d)

    def _mapping_settings(self) -> dict:
        sysd = None
        if self.systematic is not None:
            sysd = {"sigma_h": self.systematic.sigma_h, "sigma_j": self.systematic.sigma_j,
                    "sigma_j_chain": self.systematic.sigma_j_chain, "seed": self.systematic.seed}
        return _json_safe({
            "dac_bits": self.dac_bits, "dac_bits_h": self.dac_bits_h, "dac_bits_j": self.dac_bits_j,
            "ice_sigma_h": self.sigma_h, "ice_sigma_j": self.sigma_j, "ice_sigma_j_chain": self.sigma_j_chain,
            "use_extended_j": self.use_extended_j, "h_range": list(self.hw.h_range), "j_range": list(self.hw.j_range),
            "chain_j_range": list(self._chain_j_range()), "per_qubit_coupling_range": self.per_qubit_range,
            "systematic_ice": sysd})

    def config_hash(self) -> str:
        """Short hash of everything that DEFINES the emulator (not of what it has done): hardware, schedule, sampler
        settings, mapping settings, descent settings, temperature. Equal hashes = same configuration."""
        samp = {k: v for k, v in self._sampler_description().items() if k not in ("runs", "last_call")}
        cfg = {"hardware": self.hw.provenance(), "schedule": self.schedule.provenance(), "sampler": samp,
               "mapping": self._mapping_settings(), "temperature_mk": self.temperature_used_mk,
               "postprocessing": {"physical": self.pp_physical.settings(), "logical": self.pp_logical.settings()}}
        return hashlib.sha1(json.dumps(_json_safe(cfg), sort_keys=True).encode()).hexdigest()[:12]

    def provenance(self) -> dict:
        """JSON-safe record of the whole emulator. The submodules report first (hardware, schedule, sampler,
        postprocessing engines, mapping settings), then the emulator adds what only it knows."""
        prov = {
            "hardware": self.hw.provenance(),
            "schedule": self.schedule.provenance(),
            "sampler": self._sampler_description(),
            "postprocessing": {"physical": self.pp_physical.describe(), "logical": self.pp_logical.describe()},
            "mapping": self._mapping_settings(),
        }
        prov["emulator"] = {
            "class": type(self).__name__, "created_utc": self._created, "seed": self.seed,
            "runs_completed": self._runs, "temperature_used_mk": self.temperature_used_mk,
            "temperature_source": self.temperature_source, "config_hash": self.config_hash(),
            "software": _software_versions(),
        }
        return _json_safe(prov)

    # ------------------------------------------------------------------ main entry
    def sample(self, logical_bqm: dimod.BinaryQuadraticModel, embedding: dict,
               chain_strength: float | None = None, prefactor: float = 1.414,
               relative_chain_multiplier: float | None = None,
               num_reads: int = 1000, reads_per_programming: int = 100,
               chain_break_method: str = "majority", keep_target: bool = False,
               postprocess: str | None = "logical", *, spin_reversal: bool = False,
               anneal_time_us: float | None = None, keep_stages: bool = False,
               include_provenance: bool = True) -> EmulationResult:
        """
        relative_chain_multiplier : if given, caps each logical variable's chain strength at this multiplier times the
                                    TOTAL |coupling| actually incident to that variable (per-variable; embed_bqm accepts
                                    a dict chain_strength). 1.0 = a chain exactly as strong as everything pulling on it.
        reads_per_programming : the control-error realisation (and the sampler call, hence beta_eff) is redrawn every
                                this many reads, mimicking re-programming of the device between batches.
        postprocess : None (default) | "physical" | "logical" | "both". Greedy steepest descent with THIS emulator's
                      engines. "physical": on the ideal embedded model before chain resolution (it also repairs broken
                      chains); "logical": on the logical model after chain resolution. The default is None so that a
                      plain run reports the annealing emulation, not the polish. result.sampleset is the chosen stage;
                      result.raw_sampleset is always the un-descended logical output.
        spin_reversal : apply a random spin-reversal (gauge) transform in every programming cycle (applied to the
                      programmed model, undone on the samples).
        anneal_time_us : passed to samplers that accept it (SAFreezeout: moves the freeze point if freeze_s is a
                      table or a callable). Raises ValueError for samplers that do not take it.
        keep_stages : also compute all four stages from the SAME raw samples and return them in result.stages.
                      Costs extra descents; the engines' cumulative statistics include them.
        include_provenance : store self.provenance() in result.info['provenance'].
        """
        t_run = time.perf_counter()
        started = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        run_index = self._runs + 1
        timing = {}
        t_mark = [time.perf_counter()]

        def lap(name):
            now = time.perf_counter()
            timing[name] = timing.get(name, 0.0) + (now - t_mark[0])
            t_mark[0] = now

        # ---------------- validation
        if postprocess not in _POSTPROCESS:
            raise ValueError("postprocess must be None, 'physical', 'logical' or 'both'")
        if chain_break_method not in _CB_METHODS:
            raise ValueError(f"chain_break_method must be one of {_CB_METHODS}, got {chain_break_method!r}")
        if int(num_reads) < 1:
            raise ValueError("num_reads must be >= 1")
        if int(reads_per_programming) < 1:
            raise ValueError("reads_per_programming must be >= 1")
        if anneal_time_us is not None and not self._sampler_takes_anneal_time:
            raise ValueError(f"sampler {self.sampler_name!r} does not take anneal_time_us")
        logical_spin = logical_bqm.change_vartype(dimod.SPIN, inplace=False)
        missing = [v for v in logical_spin.variables if v not in embedding]
        if missing:
            raise ValueError(f"embedding has no chain for logical variables {missing[:5]}")
        bad_q = [q for ch in embedding.values() for q in ch if q not in self.hw.graph]
        if bad_q:
            raise ValueError(f"embedding uses qubits that are not in the hardware graph: {bad_q[:5]}")
        rng_state_before = copy.deepcopy(self.rng.bit_generator.state)

        # ---------------- chain strength
        raw_cs = None                                  # uncapped uniform-torque-compensation value, if computed
        if chain_strength is None:
            raw_cs = float(uniform_torque_compensation(logical_spin, embedding=embedding, prefactor=prefactor))
            if relative_chain_multiplier is not None:
                cap = self._local_chain_cap(logical_spin, embedding, relative_chain_multiplier)
                chain_strength = {v: min(raw_cs, c) for v, c in cap.items()}
            else:
                chain_strength = raw_cs

        # ---------------- 1) embed: ONE Hamiltonian from here on (sampled, post-processed and scored)
        embedded = embed_bqm(logical_spin, embedding, self.hw.graph, chain_strength=chain_strength)
        ideal = IsingArrays.from_bqm(embedded)
        qubit_to_var = {q: v for v, chain in embedding.items() for q in chain}
        owner = np.array([qubit_to_var[lbl] for lbl in ideal.labels], dtype=object)
        is_chain = owner[ideal.rows] == owner[ideal.cols]      # same chain <=> chain coupler (exactly)
        lap("embed_s")

        # ---------------- 2) map to device: scale, quantise
        chain_j_range = self._chain_j_range()
        scaled, factor = scale_to_hardware(ideal, self.hw.h_range, self.hw.j_range, is_chain=is_chain,
                                           chain_j_range=chain_j_range, per_qubit_coupling_range=self.per_qubit_range)
        programmed = quantise(scaled, self.hw.h_range, self.hw.j_range, self.dac_bits, is_chain=is_chain,
                              chain_j_range=chain_j_range, bits_h=self.dac_bits_h, bits_j=self.dac_bits_j)
        bh = self.dac_bits_h if self.dac_bits_h is not None else self.dac_bits
        bj = self.dac_bits_j if self.dac_bits_j is not None else self.dac_bits
        mapping_diag = mapping_diagnostics(ideal, scaled, programmed, self.hw.h_range, self.hw.j_range,
                                           is_chain=is_chain, chain_j_range=chain_j_range, bits_h=bh, bits_j=bj,
                                           per_qubit_coupling_range=self.per_qubit_range,
                                           sigma_h=self.sigma_h or None, sigma_j=self.sigma_j or None)
        thermal = None
        thermal_fn = getattr(self.sampler, "thermal_resolution", None)
        if callable(thermal_fn):
            try:
                thermal = thermal_fn(programmed)
            except Exception as exc:                  # for example a freeze point that still needs an anneal time
                thermal = {"error": f"{type(exc).__name__}: {exc}"}
        lap("map_s")

        # ---------------- 3) sample, one programming cycle per batch
        n_spins = len(ideal.labels)
        batches, cycles, done = [], [], 0
        while done < num_reads:
            r = min(int(reads_per_programming), int(num_reads) - done)
            g = random_gauge(n_spins, self.rng) if spin_reversal else None
            prog = apply_gauge(programmed, g) if g is not None else programmed
            noisy = add_ice_noise(prog, self.sigma_h, self.sigma_j, self.rng, is_chain=is_chain,
                                  sigma_j_chain=self.sigma_j_chain, systematic=self.systematic)
            s_seed = int(self.rng.integers(2 ** 31 - 1))
            if anneal_time_us is not None:
                X = self.sampler.sample(noisy, r, seed=s_seed, anneal_time_us=anneal_time_us)
            else:
                X = self.sampler.sample(noisy, r, seed=s_seed)
            if g is not None:
                X = undo_gauge(X, g)
            batches.append(X)
            rec = {"cycle": len(cycles), "reads": r, "sampler_seed": s_seed, "gauge": bool(g is not None)}
            li = getattr(self.sampler, "last_info", None)
            if isinstance(li, dict):
                rec.update({k: li[k] for k in _CYCLE_KEYS if k in li})
            if len(cycles) < 2000:
                cycles.append(_json_safe(rec))
            done += r
        samples = np.vstack(batches)
        lap("sampling_s")

        # ---------------- 4) stages, all from the same raw samples
        target_raw = self._target(samples, ideal, embedded)
        logical_raw = self._resolve_chains(target_raw, embedding, logical_spin, chain_break_method)
        lap("chain_resolution_s")

        stages_spin = {"raw": logical_raw}
        stage_infos = {}
        need_phys = postprocess in ("physical", "both") or keep_stages
        target_phys, logical_phys, flips_phys = target_raw, logical_raw, 0
        if need_phys:
            samples_phys, flips_phys, phys_info = self.pp_physical.run(ideal, samples)
            lap("physical_descent_s")
            target_phys = self._target(samples_phys, ideal, embedded)
            logical_phys = self._resolve_chains(target_phys, embedding, logical_spin, chain_break_method)
            lap("chain_resolution_s")
            stages_spin["physical"] = logical_phys
            stage_infos["physical"] = phys_info
        flips_stage = {"physical": int(flips_phys), "logical": 0, "both": 0}
        if postprocess == "logical" or keep_stages:
            stages_spin["logical"], flips_stage["logical"], stage_infos["logical"] = \
                self._descend_logical(logical_raw, logical_spin)
            lap("logical_descent_s")
        if postprocess == "both" or keep_stages:
            stages_spin["both"], flips_stage["both"], stage_infos["both"] = \
                self._descend_logical(logical_phys, logical_spin)
            lap("logical_descent_s")

        selected = {None: "raw", "physical": "physical", "logical": "logical", "both": "both"}[postprocess]
        final = stages_spin[selected].change_vartype(dimod.BINARY, inplace=False)
        raw_binary = logical_raw.change_vartype(dimod.BINARY, inplace=False)
        stages_out = ({k: v.change_vartype(dimod.BINARY, inplace=False) for k, v in stages_spin.items()}
                      if keep_stages else None)
        flips_phys_used = flips_stage["physical"] if postprocess in ("physical", "both") else 0
        flips_log_used = flips_stage["logical"] if postprocess == "logical" else (
            flips_stage["both"] if postprocess == "both" else 0)
        target_out = (target_phys if postprocess in ("physical", "both") else target_raw) if keep_target else None

        # ---------------- chain breaks BEFORE any descent (and after the physical one, for comparison)
        cb_raw = np.asarray(logical_raw.record["chain_break_fraction"], dtype=float)
        chain_breaks = {"raw_mean": _mean(cb_raw),
                        "raw_reads_with_a_break_pct": float(100.0 * np.mean(cb_raw > 0)) if cb_raw.size else None}
        if need_phys:
            cb_phys = np.asarray(logical_phys.record["chain_break_fraction"], dtype=float)
            chain_breaks["after_physical_descent_mean"] = _mean(cb_phys)
        lap("bookkeeping_s")

        # ---------------- the run record
        capped = isinstance(chain_strength, dict)
        cs_range = ((min(chain_strength.values()), max(chain_strength.values())) if capped
                    else (chain_strength, chain_strength))
        info = {
            # keys of the earlier version
            "sampler": self.sampler_name, "dac_bits": self.dac_bits, "sigma_h": self.sigma_h, "sigma_j": self.sigma_j,
            "num_reads": num_reads, "reads_per_programming": reads_per_programming,
            "chain_break_method": chain_break_method, "postprocess": postprocess,
            "postprocess_flips": (flips_phys_used, flips_log_used),
            "placeholder_schedule": self.schedule.is_placeholder, "chain_strength_capped": capped,
            "chain_strength_range": cs_range,
            # new
            "dac_bits_h": self.dac_bits_h, "dac_bits_j": self.dac_bits_j, "sigma_j_chain": self.sigma_j_chain,
            "spin_reversal": bool(spin_reversal), "anneal_time_us": anneal_time_us,
            "temperature_used_mk": self.temperature_used_mk, "temperature_source": self.temperature_source,
            "selected_stage": selected, "scale_factor": float(factor),
            "mapping": _json_safe(mapping_diag),
            "thermal_resolution": _json_safe(thermal),
            "embedding": _json_safe(embedding_footprint(embedding, self.hw.graph)),
            "chain_breaks": chain_breaks,
            "stages": {k: self._stage_summary(v) for k, v in stages_spin.items()},
            "postprocess_runs": _json_safe({k: v for k, v in stage_infos.items()
                                            if keep_stages or k == "physical" and postprocess in ("physical", "both")
                                            or k == selected}),
        }
        total = time.perf_counter() - t_run
        timing["total_s"] = total
        info["run"] = _json_safe({
            "index": run_index, "started_utc": started, "emulator_seed": self.seed, "rng_state_before": rng_state_before,
            "timing_s": timing, "n_programming_cycles": len(cycles), "cycles": cycles,
            "n_physical_qubits": n_spins, "n_chain_couplers": int(is_chain.sum())})
        self._runs = run_index
        if include_provenance:
            info["provenance"] = self.provenance()

        # EmulationResult.chain_strength stays a single float for backward-compat summaries (e.g.
        # scan_chain_strength): the raw uncapped uniform-torque-compensation value if we computed one
        # (auto chain strength, capped or not), otherwise whatever the caller passed in directly.
        reported_cs = raw_cs if raw_cs is not None else chain_strength
        return EmulationResult(final, reported_cs, factor, embedding_stats(embedding), target_out, info,
                               raw_binary, stages_out)


# =========================================================================
# Execution Block: Self-Test (EmulatedQPU)
# =========================================================================
if __name__ == "__main__":
    import itertools
    import json
    import math
    import sys
    import time
    import traceback
    import warnings
    from dataclasses import replace as dc_replace

    from solvers.simulated_quantum_annealing import s4_postprocess as PPM
    from solvers.simulated_quantum_annealing.s1b_embedding import find_embeddings, select_best

    QUICK = False
    NO_OPF = False
    NO_ABL = False
    NO_QUANT = False
    OUT_JSON, OUT_CASE5 = "emulator_selftest.json", "emulator_case5_result.json"
    CHECKS, REPORT, S = [], {}, {}

    # ------------------------------------------------------------------ helpers
    def check(name, ok, detail=""):
        CHECKS.append((name, bool(ok), str(detail)))
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    def info_(msg):
        print(f"    [info] {msg}")

    def section(title):
        print(f"\n{'=' * 96}\n{title}\n{'=' * 96}")

    def guarded(title, fn):
        section(title)
        try:
            fn()
        except Exception as exc:
            last = traceback.format_exc().strip().splitlines()[-1]
            check(f"section ran to the end: {title}", False, f"{type(exc).__name__}: {exc} | {last}")

    def raises(exc, fn):
        try:
            fn()
        except exc:
            return True
        except Exception:
            return False
        return False

    def jsonable(o):
        try:
            json.dumps(o)
            return True
        except Exception:
            return False

    def mat(ss):
        """(labels sorted by str, samples aligned to them): comparisons must not depend on variable order."""
        labels = sorted(ss.variables, key=str)
        pos = {v: i for i, v in enumerate(ss.variables)}
        arr = np.asarray(ss.record.sample)[:, [pos[v] for v in labels]] if len(ss) else np.zeros((0, len(labels)), dtype=np.int8)
        return labels, arr

    def same_samples(a, b):
        la, xa = mat(a)
        lb, xb = mat(b)
        return la == lb and xa.shape == xb.shape and bool(np.array_equal(xa, xb))

    def energies_of(ss, bqm):
        return np.asarray(bqm.energies(ss), dtype=float) if len(ss) else np.zeros(0)

    def cb_recompute(target_ss, embedding):
        """Chain-break fraction per read, recomputed from physical samples without Ocean."""
        pos = {v: i for i, v in enumerate(target_ss.variables)}
        X = np.asarray(target_ss.record.sample)
        frac = np.zeros(X.shape[0])
        for v, chain in embedding.items():
            sub = X[:, [pos[q] for q in chain]]
            frac += ~np.all(sub == sub[:, :1], axis=1)
        return frac / len(embedding)

    def wilson(k, n, z=1.96):
        if n == 0:
            return (float("nan"), float("nan"))
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (100 * (c - h), 100 * (c + h))

    def trim(o, n=3):
        if isinstance(o, dict):
            return {k: trim(v, n) for k, v in o.items() if k != "rng_state_before"}
        if isinstance(o, list):
            return [trim(v, n) for v in o[:n]] + ([f"... ({len(o) - n} more)"] if len(o) > n else [])
        if isinstance(o, float):
            return float(f"{o:.6g}")
        if isinstance(o, str) and len(o) > 110:
            return o[:110] + "..."
        return o

    # ------------------------------------------------------------------ setup
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SCHED = Schedule.from_csv(strict=False)
        PLACEHOLDER = Schedule.placeholder()
    print("=== Testing EmulatedQPU (every constructor / sample setting, reproducibility, provenance, case5 record) ===")
    print(f"mode={'quick' if QUICK else 'full'} | opf={'off' if NO_OPF else 'on'} | quantum samplers={'off' if NO_QUANT else 'on'}")
    print(f"schedule: {getattr(SCHED, 'source', '?')} (placeholder={getattr(SCHED, 'is_placeholder', '?')})")
    if getattr(SCHED, "is_placeholder", False):
        print("[!] placeholder schedule: the checks are internally consistent but the physical numbers are not meaningful")

    HW4 = Hardware.ideal_zephyr(m=4, t=4)
    rr0 = np.random.default_rng(3)
    nL = 8
    LBQM = dimod.BinaryQuadraticModel({i: float(rr0.uniform(-1, 1)) for i in range(nL)},
                                      {(i, j): float(rr0.uniform(-1, 1)) for i, j in itertools.combinations(range(nL), 2)},
                                      0.0, dimod.BINARY)
    LSPIN = LBQM.change_vartype(dimod.SPIN, inplace=False)
    embs = find_embeddings(LBQM, HW4, methods=("clique", "minorminer"), seeds=(0,), timeout=20.0, verbose=False)
    if not embs:
        print("[-] no embedding for the synthetic problem; cannot continue")
        sys.exit(1)
    ELAB, EMB = select_best(embs)

    re_ = np.random.default_rng(5)
    EASY = dimod.BinaryQuadraticModel({i: float(3.0 * (-1) ** i) for i in range(6)},
                                      {(i, j): float(re_.uniform(-0.2, 0.2)) for i, j in itertools.combinations(range(6), 2)},
                                      0.0, dimod.BINARY)
    ST6 = np.array(list(itertools.product((0, 1), repeat=6)), dtype=np.int8)
    E6 = EASY.energies((ST6, list(range(6))))
    GS = {i: int(ST6[int(np.argmin(E6))][i]) for i in range(6)}
    easy_embs = find_embeddings(EASY, HW4, methods=("clique",), verbose=False)
    EASY_EMB = select_best(easy_embs)[1] if easy_embs else None

    BASE_CTOR = dict(sampler="sa", sampler_kwargs=dict(num_sweeps=100), dac_bits=8, ice_sigma_h=0.005,
                     ice_sigma_j=0.002, schedule=SCHED)
    BASE_SMP = dict(num_reads=60, reads_per_programming=20, chain_break_method="majority", postprocess=None)
    HOT_KW = dict(method="pt", beta_override=0.6, pt_sweeps=30, pt_replicas=4, beta_sigma_rel=0.1)

    def make(hw=None, seed=7, **ctor):
        return EmulatedQPU(HW4 if hw is None else hw, seed=seed, **{**BASE_CTOR, **ctor})

    def go(q, bqm=None, emb=None, **smp):
        return q.sample(LBQM if bqm is None else bqm, EMB if emb is None else emb, **{**BASE_SMP, **smp})

    def valid_result(r, bqm=None):
        bqm = LBQM if bqm is None else bqm
        ss = r.sampleset
        ok = (isinstance(r, EmulationResult) and ss.vartype is dimod.BINARY and set(ss.variables) == set(bqm.variables)
              and "chain_break_fraction" in ss.record.dtype.names)
        if len(ss):
            ok = ok and bool(np.allclose(ss.record.energy, energies_of(ss, bqm), rtol=1e-9, atol=1e-6))
        return bool(ok)

    print(f"synthetic problem: {nL} dense variables, embedding {ELAB} ({sum(len(c) for c in EMB.values())} physical qubits) on {HW4.name}")

    # ================================================================== [1]
    def s1():
        sg = inspect.signature(EmulatedQPU.sample)
        dflt = {k: v.default for k, v in sg.parameters.items() if v.default is not inspect.Parameter.empty}
        info_("sample() defaults: " + ", ".join(f"{k}={dflt[k]!r}" for k in ("postprocess", "reads_per_programming", "num_reads",
              "chain_break_method", "spin_reversal", "keep_stages", "include_provenance") if k in dflt))
        q = EmulatedQPU(HW4, schedule=SCHED)
        check("default construction: physical temperature used and recorded as such",
              q.temperature_source == "physical" and q.sampler.temperature_mk == HW4.temperature_mk,
              f"{q.temperature_used_mk} mK ({q.temperature_source})")
        hw_eff = dc_replace(HW4, effective_temperature_mk=20.0)
        qe = make(hw=hw_eff)
        check("a fitted effective temperature on the hardware is used and recorded",
              qe.temperature_source == "effective" and qe.sampler.temperature_mk == 20.0)
        qo = make(hw=hw_eff, temperature_mk=30.0)
        check("temperature_mk overrides both", qo.temperature_source == "override" and qo.sampler.temperature_mk == 30.0)
        check("schedule object is used as given", make().schedule is SCHED)
        check("a missing schedule file raises when strict (default)",
              raises(FileNotFoundError, lambda: EmulatedQPU(HW4, schedule_path="/nonexistent/schedule.csv")))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qn = EmulatedQPU(HW4, schedule_path="/nonexistent/schedule.csv", strict_schedule=False)
        check("strict_schedule=False falls back to the placeholder and says so", qn.schedule.is_placeholder is True)
        try:
            qd = EmulatedQPU(HW4)
            check("default schedule path loads", not qd.schedule.is_placeholder, qd.schedule.source)
        except Exception as exc:
            info_(f"default schedule path not available here ({type(exc).__name__}); skipped")
        check("unknown sampler raises ValueError", raises(ValueError, lambda: make(sampler="nope")))
        check("systematic must be a SystematicICE (TypeError)", raises(TypeError, lambda: make(systematic="nope")))
        check("unknown postprocess setting raises TypeError", raises(TypeError, lambda: make(postprocess_settings={"nope": 1})))
        qp = make(postprocess_settings={"tie_break": "random", "seed": 3, "two_flip": True})
        pp = qp.provenance()["postprocessing"]
        check("postprocess settings reach both descent engines",
              pp["physical"]["settings"]["tie_break"] == "random" and pp["logical"]["settings"]["two_flip"] is True
              and pp["logical"]["settings"]["seed"] == 3)
        check("per_qubit_coupling_range='from_hardware' without properties raises ValueError",
              raises(ValueError, lambda: make(per_qubit_coupling_range="from_hardware")))
        hw_prop = dc_replace(HW4, properties={"per_qubit_coupling_range": [-18.0, 15.0]})
        check("per_qubit_coupling_range='from_hardware' reads the solver properties",
              make(hw=hw_prop, per_qubit_coupling_range="from_hardware").per_qubit_range == (-18.0, 15.0))
        check("the SA sampler takes anneal_time_us, SVMC does not",
              make()._sampler_takes_anneal_time is True and make(sampler="svmc", sampler_kwargs=dict(num_sweeps=5))._sampler_takes_anneal_time is False)

    # ================================================================== [2]
    def s2():
        q = make()
        st0 = copy.deepcopy(q.rng.bit_generator.state)
        bad_emb = {k: v for k, v in EMB.items() if k != 0}
        bad_q = dict(EMB)
        bad_q[0] = [10 ** 9]
        fails = {
            "postprocess='nope'": lambda: go(q, postprocess="nope"),
            "chain_break_method='nope'": lambda: go(q, chain_break_method="nope"),
            "num_reads=0": lambda: go(q, num_reads=0),
            "reads_per_programming=0": lambda: go(q, reads_per_programming=0),
            "embedding without a chain for a variable": lambda: go(q, emb=bad_emb),
            "embedding with a qubit outside the graph": lambda: go(q, emb=bad_q),
        }
        for name, fn in fails.items():
            check(f"invalid input raises ValueError: {name}", raises(ValueError, fn))
        check("anneal_time_us with a sampler that does not take it raises ValueError",
              raises(ValueError, lambda: go(make(sampler="svmc", sampler_kwargs=dict(num_sweeps=5)), anneal_time_us=1.0)))
        check("failed calls consumed no random numbers and did not count as runs",
              q._runs == 0 and q.rng.bit_generator.state == st0)

    # ================================================================== [3]
    def s3():
        q = make()
        t0 = time.time()
        r = go(q)
        dt = time.time() - t0
        S["base"], S["base_q"] = r, q
        ssl = r.sampleset
        check("result is a valid EmulationResult (BINARY, right variables, energies equal the QUBO energies)", valid_result(r))
        check("one row per read with majority vote", len(ssl) == 60 and int(np.sum(ssl.record.num_occurrences)) == 60)
        check("raw_sampleset present and identical to sampleset when postprocess is None",
              r.raw_sampleset is not None and same_samples(r.raw_sampleset, ssl))
        check("stages is None and target_sampleset is None unless requested", r.stages is None and r.target_sampleset is None)
        check("embedding_stats of the result equals embedding_stats(embedding)", r.embedding_stats == embedding_stats(EMB))
        ref_cs = float(uniform_torque_compensation(LSPIN, embedding=EMB, prefactor=1.414))
        check("automatic chain strength equals an independent uniform-torque-compensation value",
              abs(r.chain_strength - ref_cs) <= 1e-12 * max(1.0, ref_cs), f"{r.chain_strength:.4f}")
        i = r.info
        check("info is strictly JSON-serialisable", jsonable(i))
        need = {"mapping", "thermal_resolution", "embedding", "chain_breaks", "stages", "postprocess_runs", "run", "provenance",
                "scale_factor", "temperature_used_mk", "temperature_source", "selected_stage", "spin_reversal"}
        check("info has the new record keys", need <= set(i), str(sorted(need - set(i))))
        check("scale factor agrees across result, info and mapping diagnostics",
              abs(r.scale_factor - i["scale_factor"]) < 1e-15 and abs(r.scale_factor - i["mapping"]["scale_factor"]) <= 1e-12 * r.scale_factor)
        check("binding constraint is one of the known limits",
              i["mapping"]["binding_constraint"] in {"h_range", "problem_J_range", "chain_J_range", "per_qubit_coupling_range"},
              str(i["mapping"]["binding_constraint"]))
        check("embedding footprint and run record agree (chain couplers, physical qubits)",
              i["embedding"]["chain_couplers"] == i["run"]["n_chain_couplers"] and i["embedding"]["physical"] == i["run"]["n_physical_qubits"],
              f"{i['embedding']['physical']} qubits, {i['embedding']['chain_couplers']} chain couplers")
        run = i["run"]
        cyc = run["cycles"]
        check("programming cycles: ceil(60 / 20) = 3, reads add up, every cycle has its own sampler seed",
              run["n_programming_cycles"] == 3 and len(cyc) == 3 and sum(c["reads"] for c in cyc) == 60
              and len({c["sampler_seed"] for c in cyc}) == 3)
        tm = run["timing_s"]
        parts = sum(v for k, v in tm.items() if k != "total_s")
        check("timing: all stages present and their sum does not exceed the total",
              {"embed_s", "map_s", "sampling_s", "chain_resolution_s", "bookkeeping_s", "total_s"} <= set(tm) and parts <= tm["total_s"] * (1 + 1e-6) + 1e-6,
              f"total {tm['total_s']:.3f}s (wall {dt:.3f}s), sampling {tm['sampling_s']:.3f}s")
        check("run record: index 1, emulator seed, started time, RNG state before the run",
              run["index"] == 1 and run["emulator_seed"] == 7 and bool(run["started_utc"]) and "bit_generator" in run["rng_state_before"])
        check("stage summaries: only 'raw' without postprocessing; no descent runs recorded",
              set(i["stages"]) == {"raw"} and i["postprocess_runs"] == {} and i["postprocess_flips"] == (0, 0))
        check("stage summary mean equals a recomputation", abs(i["stages"]["raw"]["mean_energy"] - float(energies_of(ssl, LBQM).mean())) < 1e-9)
        check("thermal resolution reported for the SA sampler and kT_scaled = 1 / beta_nominal",
              isinstance(i["thermal_resolution"], dict) and "kT_scaled" in i["thermal_resolution"]
              and abs(i["thermal_resolution"]["kT_scaled"] - 1.0 / q.sampler.beta_final()) < 1e-9,
              f"kT_scaled {i['thermal_resolution'].get('kT_scaled')}")
        check("provenance snapshot is the emulator's provenance at the end of the run",
              i["provenance"]["emulator"]["runs_completed"] == 1 and i["provenance"]["emulator"]["config_hash"] == q.config_hash())
        r2 = go(q)
        check("a second run on the same emulator gets run index 2 and different random draws",
              r2.info["run"]["index"] == 2 and r2.info["run"]["rng_state_before"] != run["rng_state_before"])
        info_(f"synthetic run: raw chain-break mean {i['chain_breaks']['raw_mean']:.4f}, scale factor {r.scale_factor:.4g}, "
              f"binding {i['mapping']['binding_constraint']}")

    # ================================================================== [4]
    def s4():
        hot = dict(sampler_kwargs=dict(HOT_KW))
        kw = dict(chain_strength=0.4, postprocess="both", keep_stages=True)
        a, b = go(make(**hot), **kw), go(make(**hot), **kw)
        check("same seed -> identical samples in every stage",
              all(same_samples(a.stages[k], b.stages[k]) for k in a.stages) and same_samples(a.sampleset, b.sampleset))
        check("same seed -> identical per-cycle sampler seeds and betas",
              [(c["sampler_seed"], c.get("beta_used")) for c in a.info["run"]["cycles"]]
              == [(c["sampler_seed"], c.get("beta_used")) for c in b.info["run"]["cycles"]])
        c = go(make(seed=8, **hot), **kw)
        check("a different seed gives different samples", not same_samples(a.sampleset, c.sampleset))
        kw_c = dict(sampler_kwargs=dict(HOT_KW), systematic=SystematicICE(0.01, 0.005, 0.02, seed=3), ice_sigma_j_chain=0.01)
        kw_s = dict(chain_strength=0.3, chain_break_method="weighted", postprocess="both", spin_reversal=True)
        r1, r2 = go(make(**kw_c), **kw_s), go(make(**kw_c), **kw_s)
        cb = r1.info["chain_breaks"]["raw_mean"]
        check("heavy chain breaks were present in the repeat test", cb > 0, f"raw chain-break mean {cb:.3f}")
        check("weighted-random chain resolution + gauge + systematic ICE + descent: identical on repeat", same_samples(r1.sampleset, r2.sampleset))
        np.random.seed(123)
        expect = np.random.rand(3)
        np.random.seed(123)
        go(make(**kw_c), **kw_s)
        got = np.random.rand(3)
        check("the global numpy random state is restored after a run", np.array_equal(expect, got))
        import random as _r
        _r.seed(5)
        e_py = _r.random()
        _r.seed(5)
        go(make(**kw_c), **kw_s)
        check("the global python random state is restored after a run", _r.random() == e_py)
        A = make(**hot)
        a1, a2 = go(A, **kw), go(A, **kw)
        A2, B2 = make(**hot), make(seed=99, postprocess_settings={"two_flip": True}, **hot)
        x1 = go(A2, **kw)
        go(B2, **kw)
        x2 = go(A2, **kw)
        check("interleaving another emulator between runs changes nothing (no shared state)",
              same_samples(a1.sampleset, x1.sampleset) and same_samples(a2.sampleset, x2.sampleset))
        pcalls = {k: A2.pp_physical.describe()["cumulative"]["calls"] for k in ("A2",)}["A2"]
        check("each emulator counts only its own descent calls", A2.pp_physical.describe()["cumulative"]["calls"] == 2
              and B2.pp_physical.describe()["cumulative"]["calls"] == 1, f"A2 physical calls {pcalls}")
        check("the emulator does not touch the module-level engines of s4_postprocess",
              all(e["cumulative"]["calls"] == 0 for e in PPM.postprocess_report()["engines"].values()))

    # ================================================================== [5]
    def matrix():
        B = S["base"]
        fB = B.scale_factor
        ktB = B.info["thermal_resolution"]["kT_scaled"]
        cases = []

        def add(label, ctor=None, smp=None, fn=None, seed=7):
            cases.append((label, dict(ctor or {}), dict(smp or {}), fn, seed))

        def mp(r, k):
            return r.info["mapping"][k]

        def cyc(r):
            return r.info["run"]["cycles"]

        add("sampler=sa (anneal)", fn=lambda r, q: [("provenance names SAFreezeout", q.provenance()["sampler"]["class"] == "SAFreezeout", "")])
        if not NO_QUANT:
            add("sampler=svmc", ctor=dict(sampler="svmc", sampler_kwargs=dict(num_sweeps=40)),
                fn=lambda r, q: [("no thermal diagnostics for SVMC", r.info["thermal_resolution"] is None, ""),
                                 ("provenance names SVMCSampler", q.provenance()["sampler"]["class"] == "SVMCSampler", "")])
            add("sampler=sqa", ctor=dict(sampler="sqa", sampler_kwargs=dict(num_sweeps=30, trotter_slices=8)),
                fn=lambda r, q: [("no thermal diagnostics for SQA", r.info["thermal_resolution"] is None, ""),
                                 ("provenance records the Trotter slices", q.provenance()["sampler"].get("P", q.provenance()["sampler"].get("trotter_slices")) in (8, None), "")])
        add("sampler=sa, method=pt, hot beta, beta_sigma_rel=0.1", ctor=dict(sampler_kwargs=dict(HOT_KW)),
            fn=lambda r, q: [("beta fluctuates across programming cycles", len({c["beta_used"] for c in cyc(r)}) > 1, f"{[round(c['beta_used'], 3) for c in cyc(r)]}"),
                             ("nominal beta is constant at 0.6", all(abs(c["beta_nominal"] - 0.6) < 1e-12 for c in cyc(r)), "")])
        add("dac_bits=None", ctor=dict(dac_bits=None),
            fn=lambda r, q: [("no quantisation steps reported", mp(r, "step_h") is None and mp(r, "step_j") is None, "")])
        add("dac_bits=5", ctor=dict(dac_bits=5),
            fn=lambda r, q: [("steps are range / 2**5 (h 8/32, J 2/32)", abs(mp(r, "step_h") - 0.25) < 1e-12 and abs(mp(r, "step_j") - 0.0625) < 1e-12, "")])
        add("dac_bits_h=8, dac_bits_j=3", ctor=dict(dac_bits_h=8, dac_bits_j=3),
            fn=lambda r, q: [("separate steps: h 8/256, J 2/8", abs(mp(r, "step_h") - 8 / 256) < 1e-12 and abs(mp(r, "step_j") - 0.25) < 1e-12, ""),
                             ("recorded in info", r.info["dac_bits_h"] == 8 and r.info["dac_bits_j"] == 3, "")])
        add("ice sigmas = 0", ctor=dict(ice_sigma_h=0.0, ice_sigma_j=0.0),
            fn=lambda r, q: [("recorded as zero", r.info["sigma_h"] == 0.0 and r.info["sigma_j"] == 0.0, "")])
        add("ice_sigma_j_chain=0.01", ctor=dict(ice_sigma_j_chain=0.01),
            fn=lambda r, q: [("recorded in info and provenance", r.info["sigma_j_chain"] == 0.01 and q.provenance()["mapping"]["ice_sigma_j_chain"] == 0.01, "")])
        add("use_extended_j=False", ctor=dict(use_extended_j=False),
            fn=lambda r, q: [("chain J range equals the ordinary J range", q.provenance()["mapping"]["chain_j_range"] == list(HW4.j_range), ""),
                             ("the extended range gives a scale factor at least as large", fB >= r.scale_factor - 1e-12, f"{fB:.4g} vs {r.scale_factor:.4g}")])
        add("systematic ICE", ctor=dict(systematic=SystematicICE(0.01, 0.005, 0.02, seed=3)),
            fn=lambda r, q: [("provenance records the systematic ICE settings", (q.provenance()["mapping"]["systematic_ice"] or {}).get("seed") == 3, ""),
                             ("configuration hash differs from the baseline", q.config_hash() != S["base_q"].config_hash(), "")])
        add("per_qubit_coupling_range=(-0.2, 0.2)", ctor=dict(per_qubit_coupling_range=(-0.2, 0.2)),
            fn=lambda r, q: [("the limit is considered and lowers the scale factor", "per_qubit_coupling_range" in mp(r, "limits_scale_factor") and r.scale_factor < fB, f"{r.scale_factor:.4g} < {fB:.4g}")])
        add("postprocess_settings two_flip + postprocess=logical", ctor=dict(postprocess_settings={"two_flip": True}), smp=dict(postprocess="logical"),
            fn=lambda r, q: [("the logical run used two_flip", r.info["postprocess_runs"]["logical"]["settings"]["two_flip"] is True, "")])
        add("temperature_mk=24", ctor=dict(temperature_mk=24.0),
            fn=lambda r, q: [("kT_scaled doubles relative to 12 mK", abs(r.info["thermal_resolution"]["kT_scaled"] / ktB - 24.0 / HW4.temperature_mk) < 1e-9, "")])
        add("hardware with effective temperature 20 mK", ctor=dict(hw=dc_replace(HW4, effective_temperature_mk=20.0)),
            fn=lambda r, q: [("effective temperature recorded and used", r.info["temperature_source"] == "effective" and r.info["temperature_used_mk"] == 20.0, "")])
        add("placeholder schedule object", ctor=dict(schedule=PLACEHOLDER),
            fn=lambda r, q: [("placeholder flag is set in info and provenance", r.info["placeholder_schedule"] is True and q.provenance()["schedule"]["is_placeholder"] is True, "")])
        add("chain_strength=1.0 (explicit)", smp=dict(chain_strength=1.0),
            fn=lambda r, q: [("used as given, not capped", r.chain_strength == 1.0 and r.info["chain_strength_range"] == (1.0, 1.0) and r.info["chain_strength_capped"] is False, "")])
        add("chain_strength=100 (explicit)", smp=dict(chain_strength=100.0), fn=lambda r, q: [])
        add("prefactor=2.828 (twice the default)", smp=dict(prefactor=2.828),
            fn=lambda r, q: [("chain strength scales with the prefactor", abs(r.chain_strength / B.chain_strength - 2.828 / 1.414) < 1e-9, f"{r.chain_strength / B.chain_strength:.4f}")])
        add("relative_chain_multiplier=0.05", smp=dict(relative_chain_multiplier=0.05),
            fn=lambda r, q: [("capped, range ordered, never above the uncapped value",
                              r.info["chain_strength_capped"] is True and r.info["chain_strength_range"][0] <= r.info["chain_strength_range"][1] <= r.chain_strength + 1e-12,
                              f"range {tuple(round(x, 4) for x in r.info['chain_strength_range'])}, uncapped {r.chain_strength:.4f}")])
        add("num_reads=7, reads_per_programming=3", smp=dict(num_reads=7, reads_per_programming=3),
            fn=lambda r, q: [("cycles of 3, 3, 1 reads", [c["reads"] for c in cyc(r)] == [3, 3, 1], ""), ("7 rows", len(r.sampleset) == 7, "")])
        add("num_reads=50, reads_per_programming=1000", smp=dict(num_reads=50, reads_per_programming=1000),
            fn=lambda r, q: [("a single cycle", [c["reads"] for c in cyc(r)] == [50], "")])
        add("num_reads=1, reads_per_programming=1", smp=dict(num_reads=1, reads_per_programming=1),
            fn=lambda r, q: [("one row, one cycle", len(r.sampleset) == 1 and r.info["run"]["n_programming_cycles"] == 1, "")])
        add("keep_target=True", smp=dict(keep_target=True),
            fn=lambda r, q: [("target is a SPIN sampleset over all physical qubits", r.target_sampleset is not None and r.target_sampleset.vartype is dimod.SPIN
                              and len(r.target_sampleset.variables) == r.info["embedding"]["physical"], "")])
        add("spin_reversal=True", smp=dict(spin_reversal=True),
            fn=lambda r, q: [("every programming cycle was gauged", all(c["gauge"] for c in cyc(r)) and r.info["spin_reversal"] is True, "")])
        add("anneal_time_us=5 (SA, constant freeze point)", smp=dict(anneal_time_us=5.0),
            fn=lambda r, q: [("recorded", r.info["anneal_time_us"] == 5.0, "")])
        add("freeze_s table, anneal_time_us=5", ctor=dict(sampler_kwargs=dict(freeze_s=[(5.0, 0.5), (50.0, 0.7)], num_sweeps=60)), smp=dict(anneal_time_us=5.0),
            fn=lambda r, q: [("freeze point 0.5 in every cycle", all(abs(c["freeze_s_used"] - 0.5) < 1e-9 for c in cyc(r)), ""),
                             ("thermal diagnostics present (value or recorded error)", isinstance(r.info["thermal_resolution"], dict), str(r.info["thermal_resolution"])[:80])])
        add("freeze_s table, anneal_time_us=50", ctor=dict(sampler_kwargs=dict(freeze_s=[(5.0, 0.5), (50.0, 0.7)], num_sweeps=60)), smp=dict(anneal_time_us=50.0),
            fn=lambda r, q: [("freeze point 0.7 in every cycle", all(abs(c["freeze_s_used"] - 0.7) < 1e-9 for c in cyc(r)), "")])
        add("include_provenance=False", smp=dict(include_provenance=False),
            fn=lambda r, q: [("no provenance snapshot in info", "provenance" not in r.info, "")])
        add("keep_stages=True (postprocess=None)", smp=dict(keep_stages=True),
            fn=lambda r, q: [("all four stages returned", r.stages is not None and set(r.stages) == {"raw", "physical", "logical", "both"}, ""),
                             ("the selected stage is raw", r.info["selected_stage"] == "raw" and same_samples(r.sampleset, r.stages["raw"]), "")])

        print(f"    {'configuration':<52}{'rows':>6}{'mean E':>10}{'CB raw':>8}{'scale':>10}{'secs':>7}")
        rows, store = [], {}
        for label, ctor, smp, fn, seed in cases:
            t0 = time.time()
            try:
                q = make(seed=seed, **ctor)
                r = go(q, **smp)
            except Exception as exc:
                check(f"{label}: runs without error", False, f"{type(exc).__name__}: {exc}")
                continue
            dt = time.time() - t0
            e = energies_of(r.sampleset, LBQM)
            cbm = r.info["chain_breaks"]["raw_mean"]
            print(f"    {label:<52}{len(r.sampleset):>6}{(float(e.mean()) if e.size else float('nan')):>10.3f}"
                  f"{(cbm if cbm is not None else float('nan')):>8.3f}{r.scale_factor:>10.3g}{dt:>7.2f}")
            check(f"{label}: valid result and strictly JSON-safe info", valid_result(r) and jsonable(r.info))
            if fn is not None:
                for name, ok, detail in fn(r, q):
                    check(f"{label}: {name}", ok, detail)
            rows.append({"configuration": label, "rows": len(r.sampleset), "mean_energy": float(e.mean()) if e.size else None,
                         "chain_break_raw": cbm, "scale_factor": r.scale_factor, "seconds": dt})
            store[label] = r
        REPORT["settings_matrix"] = rows
        c1 = go(make(), chain_strength=1.0)
        c100 = store.get("chain_strength=100 (explicit)")
        if c100 is not None:
            check("a much larger chain strength lowers the scale factor (chain J binds)", c100.scale_factor < c1.scale_factor, f"{c100.scale_factor:.4g} < {c1.scale_factor:.4g}")
        a5, a50 = store.get("freeze_s table, anneal_time_us=5"), store.get("freeze_s table, anneal_time_us=50")
        if a5 is not None and a50 is not None and SCHED.at(0.7)[1] > SCHED.at(0.5)[1]:
            check("a longer anneal time (later freeze point) gives a larger beta",
                  a50.info["run"]["cycles"][0]["beta_used"] > a5.info["run"]["cycles"][0]["beta_used"],
                  f"{a5.info['run']['cycles'][0]['beta_used']:.3f} -> {a50.info['run']['cycles'][0]['beta_used']:.3f}")
        check("a table freeze point without anneal_time_us raises ValueError at sampling",
              raises(ValueError, lambda: go(make(sampler_kwargs=dict(freeze_s=[(5.0, 0.5), (50.0, 0.7)], num_sweeps=30)))))
        if a5 is not None and isinstance(a5.info["thermal_resolution"], dict) and "error" in a5.info["thermal_resolution"]:
            info_("the thermal diagnostic is evaluated WITHOUT the anneal time, so it reports an error for table / callable freeze points: " + a5.info["thermal_resolution"]["error"][:90])
        q = make(sampler_kwargs=dict(num_sweeps=100))
        r_a = go(q, chain_strength=0.5, relative_chain_multiplier=0.01)
        info_(f"explicit chain_strength together with relative_chain_multiplier: capped={r_a.info['chain_strength_capped']} "
              f"(the explicit value takes precedence in the current code)")

    # ================================================================== [6]
    def s6():
        if EASY_EMB is None:
            info_("no embedding for the easy problem; skipped")
            return
        info_(f"easy 6-variable problem with a unique ground state {''.join(str(GS[i]) for i in range(6))}, cold sampler, no noise, no DAC")
        cold = dict(sampler_kwargs=dict(beta_override=80.0, num_sweeps=300), ice_sigma_h=0.0, ice_sigma_j=0.0, dac_bits=None)
        gsv = None
        for name, extra in (("plain", {}), ("with spin reversal", dict(spin_reversal=True)),
                            ("spin reversal + weak systematic ICE", dict(spin_reversal=True, systematic=SystematicICE(0.002, 0.001, 0.001, seed=2)))):
            ctor = {**cold, **({"systematic": extra["systematic"]} if "systematic" in extra else {})}
            q = make(**ctor)
            r = q.sample(EASY, EASY_EMB, num_reads=60, reads_per_programming=20, spin_reversal=bool(extra.get("spin_reversal")), postprocess=None)
            labels, X = mat(r.sampleset)
            gsv = np.array([GS[v] for v in labels])
            frac = float(np.mean(np.all(X == gsv, axis=1)))
            check(f"cold sampler finds the exact ground state in >= 90% of reads: {name}", frac >= 0.9, f"{100 * frac:.1f}%")
            check(f"gauge flags match the setting: {name}", all(c["gauge"] for c in r.info["run"]["cycles"]) == bool(extra.get("spin_reversal")))

    # ================================================================== [7]
    def s7():
        hot = dict(sampler_kwargs=dict(HOT_KW))
        print(f"    {'chain_break_method':<20}{'rows':>6}{'CB raw':>9}{'mean E':>10}")
        for m in ("majority", "weighted", "minimize_energy", "discard"):
            r = go(make(**hot), chain_strength=0.5, chain_break_method=m)
            e = energies_of(r.sampleset, LBQM)
            print(f"    {m:<20}{len(r.sampleset):>6}{r.info['chain_breaks']['raw_mean']:>9.3f}{(float(e.mean()) if e.size else float('nan')):>10.3f}")
            check(f"{m}: valid result", valid_result(r))
            if m == "discard":
                cbf = np.asarray(r.sampleset.record["chain_break_fraction"]) if len(r.sampleset) else np.zeros(0)
                check("discard keeps at most the reads and only unbroken ones", len(r.sampleset) <= 60 and bool(np.all(cbf == 0)), f"{len(r.sampleset)} of 60 kept")
            else:
                check(f"{m}: one row per read", len(r.sampleset) == 60)
        r = go(make(**hot), chain_strength=0.3, postprocess=None, keep_target=True)
        per_row = cb_recompute(r.target_sampleset, EMB)
        check("the raw chain-break fraction equals an independent recomputation, read by read",
              np.allclose(per_row, np.asarray(r.raw_sampleset.record["chain_break_fraction"]), atol=1e-12),
              f"mean {per_row.mean():.4f} vs reported {r.info['chain_breaks']['raw_mean']:.4f}")
        check("chain_breaks.raw_mean matches", abs(per_row.mean() - r.info["chain_breaks"]["raw_mean"]) < 1e-12)
        r2 = go(make(**hot), chain_strength=0.3, postprocess="physical", keep_target=True, keep_stages=True)
        per_row_p = cb_recompute(r2.target_sampleset, EMB)
        check("after physical descent: reported fraction equals a recomputation on the descended physical samples",
              np.allclose(per_row_p, np.asarray(r2.stages["physical"].record["chain_break_fraction"]), atol=1e-12)
              and abs(per_row_p.mean() - r2.info["chain_breaks"]["after_physical_descent_mean"]) < 1e-12)
        info_(f"chain breaks: raw {r2.info['chain_breaks']['raw_mean']:.4f} -> after physical descent {r2.info['chain_breaks']['after_physical_descent_mean']:.4f} "
              f"(with weak chains the descent can also break chains; it is not a pure repair)")

    # ================================================================== [8]
    def s8():
        hot = dict(sampler_kwargs=dict(HOT_KW))
        kw = dict(chain_strength=0.3)
        tol = 1e-9
        for mode in (None, "physical", "logical", "both"):
            qa, qb = make(**hot), make(**hot)
            a = go(qa, postprocess=mode, **kw)
            b = go(qb, postprocess=mode, keep_stages=True, **kw)
            sel = {None: "raw", "physical": "physical", "logical": "logical", "both": "both"}[mode]
            check(f"postprocess={mode}: the same seed gives the same result with and without keep_stages", same_samples(a.sampleset, b.sampleset))
            check(f"postprocess={mode}: the selected stage is '{sel}' and equals stages['{sel}']", b.info["selected_stage"] == sel and same_samples(b.sampleset, b.stages[sel]))
            check(f"postprocess={mode}: raw_sampleset equals stages['raw']", same_samples(b.raw_sampleset, b.stages["raw"]))
            fl = a.info["postprocess_flips"]
            run_ = a.info["postprocess_runs"]
            exp_phys = run_["physical"]["flips"] if mode in ("physical", "both") else 0
            exp_log = run_[sel]["flips"] if mode in ("logical", "both") else 0
            check(f"postprocess={mode}: flips tuple equals the engines' reports", fl == (exp_phys, exp_log), f"{fl}")
            check(f"postprocess={mode}: descent runs recorded for the stages used",
                  set(run_) == {None: set(), "physical": {"physical"}, "logical": {"logical"}, "both": {"physical", "both"}}[mode], str(sorted(run_)))
            check(f"postprocess={mode}: JSON-safe", jsonable(a.info) and jsonable(b.info))
        qk = make(**hot)
        r = go(qk, postprocess=None, keep_stages=True, **kw)
        eraw, ephy = energies_of(r.stages["raw"], LBQM), energies_of(r.stages["physical"], LBQM)
        elog, eboth = energies_of(r.stages["logical"], LBQM), energies_of(r.stages["both"], LBQM)
        check("all stages have one row per read", all(len(v) == 60 for v in r.stages.values()))
        check("logical descent never raises the logical energy of a read", bool(np.all(elog <= eraw + tol)), f"mean {eraw.mean():.4f} -> {elog.mean():.4f}")
        check("'both' is never worse than the physically descended stage", bool(np.all(eboth <= ephy + tol)), f"mean {ephy.mean():.4f} -> {eboth.mean():.4f}")
        for k, v in r.stages.items():
            e = energies_of(v, LBQM)
            check(f"stage summary for '{k}' equals a recomputation", abs(r.info["stages"][k]["mean_energy"] - float(e.mean())) < 1e-9)
        check("engine counters: physical engine 1 call, logical engine 2 calls (stage 'logical' and 'both')",
              qk.pp_physical.describe()["cumulative"]["calls"] == 1 and qk.pp_logical.describe()["cumulative"]["calls"] == 2)
        check("every descent run reports zero energy increases",
              all(v["energy_increased_reads"] == 0 for v in r.info["postprocess_runs"].values()))
        info_(f"mean logical energy by stage: raw {eraw.mean():.4f} | physical {ephy.mean():.4f} | logical {elog.mean():.4f} | both {eboth.mean():.4f}")
        REPORT["stage_energies"] = {"raw": float(eraw.mean()), "physical": float(ephy.mean()), "logical": float(elog.mean()), "both": float(eboth.mean())}

    # ================================================================== [9]
    def s9():
        q = make()
        h0 = q.config_hash()
        pv = q.provenance()
        check("provenance sections: hardware, schedule, sampler, postprocessing (physical + logical), mapping, emulator",
              {"hardware", "schedule", "sampler", "postprocessing", "mapping", "emulator"} <= set(pv) and {"physical", "logical"} <= set(pv["postprocessing"]))
        check("strictly JSON-safe", jsonable(pv))
        em = pv["emulator"]
        check("emulator section: seed, creation time, runs, temperature and source, hash, software",
              em["seed"] == 7 and bool(em["created_utc"]) and em["runs_completed"] == 0 and em["temperature_used_mk"] == 12.0
              and em["temperature_source"] == "physical" and len(em["config_hash"]) == 12 and {"numpy", "dimod"} <= set(em["software"]),
              f"hash {em['config_hash']}")
        check("hardware and schedule sections come from the submodules' own provenance methods",
              pv["hardware"] == HW4.provenance() and pv["schedule"] == SCHED.provenance())
        check("same configuration -> same hash (also across seeds: the seed is not part of the configuration)",
              make().config_hash() == h0 and make(seed=99).config_hash() == h0)
        variants = {"dac_bits": dict(dac_bits=5), "sampler settings": dict(sampler_kwargs=dict(num_sweeps=200)),
                    "temperature": dict(temperature_mk=15.0), "descent settings": dict(postprocess_settings={"two_flip": True}),
                    "schedule": dict(schedule=PLACEHOLDER), "systematic ICE": dict(systematic=SystematicICE(0.01, 0.005, seed=1)),
                    "per-qubit limit": dict(per_qubit_coupling_range=(-18.0, 15.0)), "ice sigma": dict(ice_sigma_h=0.03),
                    "chain sigma": dict(ice_sigma_j_chain=0.01), "extended J": dict(use_extended_j=False)}
        bad = [k for k, kw in variants.items() if make(**kw).config_hash() == h0]
        check("every setting that defines the emulator changes the hash", not bad, f"unchanged: {bad}")
        r = go(q)
        check("running does not change the hash; runs_completed advances",
              q.config_hash() == h0 and q.provenance()["emulator"]["runs_completed"] == 1)
        check("the sampler record counts the programming cycles of the run",
              q.provenance()["sampler"].get("runs", {}).get("programming_cycles") == 3, str(q.provenance()["sampler"].get("runs")))
        check("physical and logical engine records exist and are independent",
              q.provenance()["postprocessing"]["physical"]["label"] == "physical" and q.provenance()["postprocessing"]["logical"]["label"] == "logical")
        check("the snapshot stored in the result matches the emulator after the run", r.info["provenance"]["emulator"]["config_hash"] == h0)

    # ================================================================== [10]
    def s10():
        try:
            import copy as _copy
            import pandapower as pp
            import pandapower.networks as nw
            from solvers.qubo_formulator import QuboFormulator
        except ImportError as exc:
            info_(f"skipped: cannot import pandapower / formulator ({exc})")
            return
        net = nw.case5()
        if not net.ext_grid.empty:
            net.ext_grid["min_p_mw"] = 0.0
        tmp = _copy.deepcopy(net)
        pp.rundcopp(tmp)
        ref = float(tmp.res_cost)
        ref_disp = {}
        for name, tbl in (("gen", "res_gen"), ("sgen", "res_sgen"), ("ext_grid", "res_ext_grid")):
            try:
                ref_disp[name] = {int(i): round(float(v), 3) for i, v in getattr(tmp, tbl)["p_mw"].items()}
            except Exception:
                pass
        fm = QuboFormulator(formulation="dc_ptdf", mw_precision=10.0, encoding="radix")
        bqm5, _ = fm._formulate_qubo(net)
        hw12 = Hardware.ideal_zephyr(m=12, t=4)
        e5 = find_embeddings(bqm5, hw12, methods=("clique", "minorminer"), seeds=(0,), timeout=30.0, verbose=False)
        if not e5:
            info_("no embedding for case5; skipped")
            return
        lab5, emb5 = select_best(e5)
        S["case5"] = dict(net=net, fm=fm, bqm=bqm5, emb=emb5, hw=hw12, ref=ref, ref_disp=ref_disp, lab=lab5, fp=e5[lab5]["footprint"], pp=pp, copy=_copy)

        def decode_set(ss, with_best=True):
            labels, k, n, best, best_pack, cache = list(ss.variables), 0, 0, None, None, {}
            for row, occ in zip(np.asarray(ss.record.sample), ss.record.num_occurrences):
                key = tuple(int(v) for v in row)
                if key not in cache:
                    cache[key] = fm._decode_solution({l: int(v) for l, v in zip(labels, row)}, net)
                out = cache[key]
                n += int(occ)
                if out[4]["is_feasible"]:
                    k += int(occ)
                    if best is None or out[3] < best:
                        best, best_pack = out[3], (out, key)
            gap = None if (best is None or not ref) else 100.0 * (best - ref) / abs(ref)
            lo, hi = wilson(k, n)
            return {"feasible_pct": 100.0 * k / n if n else None, "feasible_ci95_pct": [lo, hi], "n_reads": n,
                    "best_feasible_cost": best, "gap_to_classical_pct": gap,
                    "mean_logical_energy": float(np.mean(ss.record.energy)) if len(ss) else None}, best_pack, cache
        S["decode_set"] = decode_set

        ctor = dict(sampler="sa", sampler_kwargs=dict(num_sweeps=300), dac_bits=8, ice_sigma_h=0.01, ice_sigma_j=0.005,
                    ice_sigma_j_chain=0.005, systematic=SystematicICE(0.003, 0.001, 0.002, seed=5), schedule=SCHED)
        smp = dict(num_reads=100 if QUICK else 200, reads_per_programming=50, chain_break_method="majority", postprocess="logical",
                   keep_stages=True, spin_reversal=True)
        q = EmulatedQPU(hw12, seed=11, **ctor)
        t0 = time.time()
        r = q.sample(bqm5, emb5, **smp)
        dt = time.time() - t0
        print(f"    case5: {len(bqm5.variables)} logical variables, embedding {lab5} ({e5[lab5]['stats']['physical']} qubits), classical DC-OPF cost {ref:.3f}; run took {dt:.1f}s")
        stages_doc, best_over = {}, None
        print(f"    {'stage':<10}{'feasible %':>12}{'95% CI':>18}{'gap %':>9}{'mean logical E':>18}")
        for k, v in r.stages.items():
            d, bp, _ = decode_set(v)
            stages_doc[k] = d
            lo, hi = d["feasible_ci95_pct"]
            print(f"    {k:<10}{d['feasible_pct']:>12.1f}{f'[{lo:.1f}, {hi:.1f}]':>18}{('-' if d['gap_to_classical_pct'] is None else format(d['gap_to_classical_pct'], '.2f')):>9}{d['mean_logical_energy']:>18.2f}")
            if k == r.info["selected_stage"]:
                best_over = (k, bp)
        check("every stage holds all reads", all(len(v) == smp["num_reads"] for v in r.stages.values()))
        check("the selected stage (logical) has feasible samples", stages_doc["logical"]["feasible_pct"] > 0, f"{stages_doc['logical']['feasible_pct']:.1f}%")
        info_(f"the raw emulator output is {stages_doc['raw']['feasible_pct']:.1f}% feasible; the postprocessed stages show what the polish adds")
        best_doc = None
        if best_over and best_over[1]:
            (disp, sgen, slack, cost, feas), key = best_over[1]
            best_doc = {"stage": best_over[0], "cost": cost, "gap_to_classical_pct": 100.0 * (cost - ref) / abs(ref),
                        "gen_dispatch_mw": disp, "sgen_dispatch_mw": sgen, "ext_grid_dispatch_mw": slack, "feasibility": feas}
            check("best feasible solution is feasible by the formulator's own decoder", bool(feas["is_feasible"]))
        sel = r.sampleset
        order = np.argsort(np.asarray(sel.record.energy))[:5]
        labels = list(sel.variables)
        top = []
        for i in order:
            row = np.asarray(sel.record.sample)[i]
            o = fm._decode_solution({l: int(v) for l, v in zip(labels, row)}, net)
            top.append({"logical_energy": float(sel.record.energy[i]), "cost": o[3], "is_feasible": bool(o[4]["is_feasible"]),
                        "raw_imbalance_mw": o[4]["raw_imbalance_mw"], "max_line_violation_mw": o[4]["max_line_violation_mw"],
                        "gen_dispatch_mw": o[0], "ext_grid_dispatch_mw": o[2]})
        doc = {"problem": {"name": "case5", "n_logical_variables": len(bqm5.variables), "n_qubo_couplings": int(bqm5.num_interactions),
                           "formulation": "dc_ptdf, radix encoding, 10 MW precision",
                           "classical_reference": {"cost": ref, "dispatch_mw": ref_disp}},
               "emulator_call": {"constructor": {"sampler": "sa", "sampler_kwargs": ctor["sampler_kwargs"], "dac_bits": ctor["dac_bits"],
                                                 "ice_sigma_h": ctor["ice_sigma_h"], "ice_sigma_j": ctor["ice_sigma_j"], "ice_sigma_j_chain": ctor["ice_sigma_j_chain"],
                                                 "systematic_ice": "SystematicICE(0.003, 0.001, 0.002, seed=5)", "seed": 11},
                                 "sample": smp},
               "embedding": {"label": lab5, "footprint": e5[lab5]["footprint"]},
               "stages": stages_doc, "selected_stage": r.info["selected_stage"], "best_solution": best_doc,
               "lowest_energy_samples_of_selected_stage": top, "run_info": r.info}
        with open(OUT_CASE5, "w") as fh:
            json.dump(_json_safe(doc), fh, indent=1)
        with open(OUT_CASE5) as fh:
            back = json.load(fh)
        check("the case5 record was written and reloads", back["problem"]["name"] == "case5" and "run_info" in back)
        check("the record carries the emulator provenance and its hash",
              back["run_info"]["provenance"]["emulator"]["config_hash"] == q.config_hash())
        prev = copy.deepcopy(_json_safe(doc))
        pv = prev["run_info"]["provenance"]
        prev["run_info"]["provenance"] = {"emulator": pv["emulator"], "sampler": pv["sampler"], "mapping": pv["mapping"],
                                          "other_sections": [k for k in pv if k not in ("emulator", "sampler", "mapping")]}
        print(f"\n    ---- preview of {OUT_CASE5} (lists cut to 3 items, long values shortened; the file has everything) ----")
        print(json.dumps(trim(prev), indent=1))
        print("    ---- end of preview ----")
        REPORT["case5_stages"] = stages_doc

    # ================================================================== [11]
    def s11():
        if "case5" not in S:
            info_("needs section 10; skipped")
            return
        C = S["case5"]
        decode_set = S["decode_set"]
        base = dict(sampler="sa", sampler_kwargs=dict(num_sweeps=300), dac_bits=8, ice_sigma_h=0.01, ice_sigma_j=0.005, schedule=SCHED)
        reads = 100 if QUICK else 200
        smp = dict(num_reads=reads, reads_per_programming=50, chain_break_method="majority", postprocess=None, keep_stages=True)
        cfgs = [("baseline (dac 8, ICE 0.01/0.005)", {}, {}),
                ("no ICE", dict(ice_sigma_h=0.0, ice_sigma_j=0.0), {}),
                ("no DAC", dict(dac_bits=None), {}),
                ("no ICE, no DAC", dict(ice_sigma_h=0.0, ice_sigma_j=0.0, dac_bits=None), {}),
                ("no ICE, no DAC, ordinary J range for chains", dict(ice_sigma_h=0.0, ice_sigma_j=0.0, dac_bits=None, use_extended_j=False), {}),
                ("no ICE, no DAC, neal auto-beta (no physical kT)", dict(ice_sigma_h=0.0, ice_sigma_j=0.0, dac_bits=None, sampler_kwargs=dict(num_sweeps=300, use_auto_beta=True)), {}),
                ("chain strength x0.5", {}, dict(prefactor=0.707)),
                ("chain strength x2", {}, dict(prefactor=2.828)),
                ("hotter sampler, 24 mK", dict(temperature_mk=24.0), {}),
                ("PT at the physical beta", dict(sampler_kwargs=dict(method="pt", pt_sweeps=200, pt_replicas=16)), {})]
        if QUICK:
            cfgs = [c for c in cfgs if c[0] in ("baseline (dac 8, ICE 0.01/0.005)", "no ICE, no DAC", "no ICE, no DAC, neal auto-beta (no physical kT)")]
        print(f"    case5, {reads} reads per configuration, same seed for all; feasibility of the RAW stage (no descent) and of the logical-descent stage")
        print(f"    {'configuration':<50}{'raw feas%':>10}{'raw 95% CI':>16}{'logical %':>10}{'scale':>10}{'J zeroed%':>10}{'kT_sc':>8}{'CB raw':>8}{'s':>6}")
        rows = []
        for label, c_over, s_over in cfgs:
            try:
                q = EmulatedQPU(C["hw"], seed=21, **{**base, **c_over})
                t0 = time.time()
                r = q.sample(C["bqm"], C["emb"], **{**smp, **s_over})
                dt = time.time() - t0
                dr, _, _ = decode_set(r.stages["raw"])
                dl, _, _ = decode_set(r.stages["logical"])
            except Exception as exc:
                check(f"ablation '{label}' ran", False, f"{type(exc).__name__}: {exc}")
                continue
            th = r.info["thermal_resolution"] or {}
            lo, hi = dr["feasible_ci95_pct"]
            kt = th.get("kT_scaled")
            print(f"    {label:<50}{dr['feasible_pct']:>10.1f}{f'[{lo:.0f}, {hi:.0f}]':>16}{dl['feasible_pct']:>10.1f}{r.scale_factor:>10.2e}"
                  f"{r.info['mapping']['problem_J_zeroed_pct']:>10.1f}{('-' if kt is None else format(kt, '.4f')):>8}{r.info['chain_breaks']['raw_mean']:>8.3f}{dt:>6.1f}")
            rows.append({"configuration": label, "raw_feasible_pct": dr["feasible_pct"], "raw_ci95": dr["feasible_ci95_pct"],
                         "logical_feasible_pct": dl["feasible_pct"], "scale_factor": r.scale_factor,
                         "problem_J_zeroed_pct": r.info["mapping"]["problem_J_zeroed_pct"], "kT_scaled": kt,
                         "chain_break_raw": r.info["chain_breaks"]["raw_mean"], "seconds": dt,
                         "binding_constraint": r.info["mapping"]["binding_constraint"]})
        REPORT["case5_ablation"] = rows
        check("the ablation produced a row for every configuration", len(rows) == len(cfgs))
        if len(rows) >= 2:
            best = max(rows, key=lambda x: x["raw_feasible_pct"])
            worst = min(rows, key=lambda x: x["raw_feasible_pct"])
            info_(f"raw feasibility ranges from {worst['raw_feasible_pct']:.1f}% ('{worst['configuration']}') to {best['raw_feasible_pct']:.1f}% ('{best['configuration']}'); "
                  f"intervals that overlap are not distinguishable at this number of reads")

    # ------------------------------------------------------------------ run
    guarded("[1] construction, temperature / schedule sources, validation of constructor arguments", s1)
    guarded("[2] sample(): invalid inputs are rejected before any work is done", s2)
    guarded("[3] a basic run: result structure, run record, timing, cycles", s3)
    guarded("[4] reproducibility and isolation (seeds, global generators, interleaved emulators)", s4)
    guarded("[5] every setting, one at a time", matrix)
    guarded("[6] correctness with a cold sampler: spin reversal and systematic ICE must not change the answer", s6)
    guarded("[7] chain-break methods and an independent recomputation of the chain-break fractions", s7)
    guarded("[8] postprocess stages: consistency, monotonicity, reports", s8)
    guarded("[9] provenance and the configuration hash", s9)
    if not NO_OPF:
        guarded("[10] DC-OPF case5: all stages, decoded solutions, the example result record", s10)
        if not NO_ABL:
            guarded("[11] case5 ablation: where does the raw feasibility go?", s11)

    # ------------------------------------------------------------------ summary
    section("SUMMARY")
    failed = [c for c in CHECKS if not c[1]]
    print(f"checks: {len(CHECKS) - len(failed)} passed, {len(failed)} failed")
    for name, _, detail in failed:
        print(f"  FAILED: {name}  {detail}")
    with open(OUT_JSON, "w") as fh:
        json.dump({"settings": {"quick": QUICK, "no_opf": NO_OPF, "no_ablation": NO_ABL, "no_quantum": NO_QUANT,
                                "schedule_source": getattr(SCHED, "source", None), "schedule_placeholder": getattr(SCHED, "is_placeholder", None)},
                   "checks": CHECKS, "report": REPORT}, fh, default=str, indent=1)
    print(f"details written to {OUT_JSON}" + ("" if NO_OPF else f"; example result record in {OUT_CASE5}"))
    sys.exit(1 if failed else 0)