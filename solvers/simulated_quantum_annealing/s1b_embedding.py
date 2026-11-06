"""Step 1b: minor embedding of the logical problem graph into the Zephyr graph.

Methods:
    clique          busclique structured clique embedding (deterministic)
    minorminer      Cai, Macready, Roy 2014 heuristic, one run per seed
    layout          layout-aware minorminer (Ocean's minorminer.layout), one run per seed
    mm_from_clique  minorminer refinement started from the busclique embedding, one run per seed

Every attempt is recorded, including failures and their reasons. Other modules can call:
    embedding_footprint(embedding, target, label)   -> JSON-safe dict for one embedding
    EmbeddingSet.report(chosen)                     -> JSON-safe dict for the whole exploration
"""
from __future__ import annotations

import time
from collections import Counter

import networkx as nx


#########################################################################################################
# Basic helpers
#########################################################################################################
def source_graph(bqm) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(bqm.variables)
    g.add_edges_from((u, v) for u, v, _ in bqm.iter_quadratic())
    return g


def embedding_stats(embedding: dict) -> dict:
    lens = [len(c) for c in embedding.values()]
    if not lens:
        return {"logical": 0, "physical": 0, "max_chain": 0, "min_chain": 0, "mean_chain": 0.0, "hist": {}}
    return {"logical": len(lens), "physical": sum(lens), "max_chain": max(lens), "min_chain": min(lens),
            "mean_chain": sum(lens) / len(lens), "hist": dict(sorted(Counter(lens).items()))}


def embedding_footprint(embedding: dict, target: nx.Graph | None = None,
                        label: str | None = None, seconds: float | None = None) -> dict:
    """JSON-safe description of one embedding, for stamping into result files from any module.
    With `target`, also reports chain couplers (hardware edges inside chains: what embed_bqm programs as
    chain couplings) and the fraction of the device used."""
    d = embedding_stats(embedding)
    d["hist"] = {str(k): v for k, v in d["hist"].items()}
    if label is not None:
        d["label"] = label
    if seconds is not None:
        d["seconds"] = round(float(seconds), 3)
    if target is not None:
        d["chain_couplers"] = sum(target.subgraph(chain).number_of_edges() for chain in embedding.values())
        d["device_qubits_used_pct"] = round(100.0 * d["physical"] / max(1, target.number_of_nodes()), 3)
    return d


def _is_valid(embedding, src: nx.Graph, tgt: nx.Graph) -> bool:
    from dwave.embedding import is_valid_embedding
    return bool(is_valid_embedding(embedding, src, tgt))


#########################################################################################################
# Embedding algorithms
#########################################################################################################
def embed_minorminer(src: nx.Graph, tgt: nx.Graph, seed: int = 0, tries: int = 20, timeout: float = 120.0):
    import minorminer
    emb = minorminer.find_embedding(src, tgt, random_seed=seed, tries=tries, timeout=timeout)
    return {v: list(c) for v, c in emb.items()} if emb else None


def embed_minorminer_seeded(src: nx.Graph, tgt: nx.Graph, initial: dict, seed: int = 0,
                            tries: int = 20, timeout: float = 120.0):
    """minorminer refinement starting from an existing embedding (e.g. the busclique clique)."""
    import minorminer
    emb = minorminer.find_embedding(src, tgt, initial_chains={v: list(c) for v, c in initial.items()},
                                    random_seed=seed, tries=tries, timeout=timeout)
    return {v: list(c) for v, c in emb.items()} if emb else None


def embed_layout(src: nx.Graph, tgt: nx.Graph, seed: int = 0, timeout: float = 120.0):
    """Layout-aware minorminer (Ocean's minorminer.layout). Needs the target graph to carry its
    dwave_networkx attributes (family/rows/tile); a Hardware.load()ed graph does not, and fails here."""
    from minorminer import layout
    emb = layout.find_embedding(src, tgt, random_seed=seed, timeout=timeout)
    return {v: list(c) for v, c in emb.items()} if emb else None


def max_embeddable_clique(tgt: nx.Graph) -> int:
    """How large a complete graph CAN possibly embed in tgt, per busclique's own cache -- answers
    "why doesn't my clique fit" in milliseconds, with no search or timeout, before trying anything
    else. Returns 0 if busclique is unavailable or the target has no cached clique capacity."""
    try:
        from minorminer.busclique import busgraph_cache
    except ImportError:
        return 0
    try:
        return len(busgraph_cache(tgt).largest_clique())
    except Exception:
        return 0


def embed_clique(src: nx.Graph, tgt: nx.Graph, verbose: bool = True):
    """Structured clique embedding (busclique): every pair of logical variables is connectable, so it
    is a safe but often chain-hungry choice for dense problems such as dc_ptdf. On failure, the real
    reason from busclique is surfaced instead of swallowed -- for a dense/near-complete source graph
    this is almost always a hard physical-qubit capacity limit (see max_embeddable_clique), not a
    transient search failure, so retrying or raising a timeout will not fix it."""
    try:
        from minorminer import busclique
    except ImportError as e:
        if verbose:
            print(f"  [clique] busclique unavailable: {e}")
        return None
    try:
        emb = busclique.find_clique_embedding(list(src.nodes), tgt)
    except Exception as e:
        if verbose:
            cap = max_embeddable_clique(tgt)
            hint = (f" (largest embeddable clique on this target: {cap} nodes, "
                    f"source needs {src.number_of_nodes()})" if cap else "")
            print(f"  [clique] busclique failed: {e}{hint}")
        return None
    return {v: list(c) for v, c in emb.items()} if emb else None


#########################################################################################################
# Method registry
#########################################################################################################
# each runner: (src, tgt, seed, timeout, found, verbose) -> embedding dict or None; may raise
def _run_clique(src, tgt, seed, timeout, found, verbose):
    emb = embed_clique(src, tgt, verbose=verbose)
    if not emb:
        cap, n = max_embeddable_clique(tgt), src.number_of_nodes()
        if cap and n > cap:
            raise RuntimeError(f"clique capacity exceeded: source has {n} nodes, "
                               f"busclique's largest clique on this target is {cap}")
    return emb


def _run_minorminer(src, tgt, seed, timeout, found, verbose):
    return embed_minorminer(src, tgt, seed=seed, timeout=timeout)


def _run_layout(src, tgt, seed, timeout, found, verbose):
    return embed_layout(src, tgt, seed=seed, timeout=timeout)


def _run_mm_from_clique(src, tgt, seed, timeout, found, verbose):
    if "clique" not in found:
        raise SkipMethod("needs a successful 'clique' embedding first (not requested, or it failed)")
    return embed_minorminer_seeded(src, tgt, found["clique"]["embedding"], seed=seed, timeout=timeout)


# name -> (runner, run_once_per_seed). To add a method: write a runner above and add one line here.
METHODS = {
    "clique": (_run_clique, False),
    "minorminer": (_run_minorminer, True),
    "layout": (_run_layout, True),
    "mm_from_clique": (_run_mm_from_clique, True),
}


class EmbeddingSet(dict):
    """Drop-in for the old {label: {"embedding", "stats", "seconds"}} dict (successes only), plus
    .attempts: one record per method/seed tried, INCLUDING failures and why. Entries also carry "footprint"."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.attempts, self.source, self.target = [], {}, ""

    def report(self, chosen: str | None = None) -> dict:
        """JSON-safe summary of the whole exploration, for result files."""
        return {"source_graph": self.source, "target": self.target, "chosen": chosen, "attempts": self.attempts}

class SkipMethod(RuntimeError):
    """Raised by a runner that cannot apply (a prerequisite is missing). Logged as skipped, not failed."""


def _provably_impossible(src: nx.Graph, tgt: nx.Graph):
    """Necessary conditions for ANY minor embedding. Returns a reason string if one is violated (a proof), else None."""
    n, m, tn, tm = src.number_of_nodes(), src.number_of_edges(), tgt.number_of_nodes(), tgt.number_of_edges()
    if n > tn:
        return f"proven infeasible: {n} source nodes > {tn} target qubits (every chain needs a qubit)"
    if m > tm:
        return (f"proven infeasible: {m} source edges > {tm} target couplers "
                f"(each source edge needs its own coupler between disjoint chains)")
    return None


#########################################################################################################
# Exploration and selection
#########################################################################################################
def find_embeddings(bqm, hardware, methods=("clique", "minorminer"), seeds=(0, 1, 2),
                    timeout: float = 36000.0, verbose: bool = True, density_skip_minorminer: float = 0.9,
                    skip_minorminer_if_clique_found: bool = False, skip_beyond_clique_capacity: bool = True) -> EmbeddingSet:
    """Tries every requested method (see METHODS) and records every outcome.

    methods : names from METHODS, run in the order given. Per-seed methods run once per seed, labelled
              "<method>_seed<k>"; clique keeps the label "clique".
    density_skip_minorminer : if the source graph's density is at or above this (e.g. your balance
              clique at ~100%), a capacity precheck runs FIRST via max_embeddable_clique -- if the
              source is provably too large to ever embed, nothing is attempted and an empty set is
              returned immediately rather than after a multi-minute timeout per seed. None disables it.
    skip_minorminer_if_clique_found : if True, once "clique" succeeds every other method is skipped.
              False (default) keeps all, e.g. for benchmarking the methods against each other.
    skip_beyond_clique_capacity : if True (default), a dense source (density >= density_skip_minorminer) with more
              nodes than busclique's clique capacity is skipped as "likely infeasible". This is a heuristic,
              not a proof. Set False to attempt it anyway. Provably impossible sources (too many nodes or
              edges for the target) are always skipped.
    A method that raises or fails is logged in .attempts with its reason; it does not stop the others.
    """
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"unknown embedding methods {unknown}; available: {list(METHODS)}")

    src, tgt = source_graph(bqm), hardware.graph
    n = src.number_of_nodes()
    out = EmbeddingSet()
    out.source = {"n_vars": n, "n_edges": src.number_of_edges(),
                  "density": round(nx.density(src), 4) if n > 1 else 0.0}
    out.target = hardware.summary()

    reason = _provably_impossible(src, tgt)
    proven = reason is not None
    if (reason is None and skip_beyond_clique_capacity and density_skip_minorminer is not None
            and n > 1 and nx.density(src) >= density_skip_minorminer):
        cap = max_embeddable_clique(tgt)
        if cap and n > cap:
            reason = (f"likely infeasible (not a proof): dense source, {n} nodes, density "
                      f"{nx.density(src):.0%}, exceeds busclique's clique capacity of {cap}")
    if reason is not None:
        out.attempts.append({"label": "precheck", "method": "precheck", "seed": None, "ok": False,
                             "skipped": True, "proven": proven, "error": reason})
        if verbose:
            print(f"source graph: {n} vars, density {nx.density(src):.2%} | {hardware.summary()}")
            print(f"  [precheck] {reason} -- skipping all methods")
        return out

    for name in methods:
        fn, per_seed = METHODS[name]
        if skip_minorminer_if_clique_found and name != "clique" and "clique" in out:
            continue
        for sd in (seeds if per_seed else (None,)):
            label = f"{name}_seed{sd}" if per_seed else name
            t0, emb, err, skipped = time.time(), None, None, False
            try:
                emb = fn(src, tgt, sd, timeout, out, verbose)
                if not emb:
                    err = "no embedding found"
                elif not _is_valid(emb, src, tgt):
                    emb, err = None, "result failed is_valid_embedding"
            except SkipMethod as e:
                emb, err, skipped = None, str(e), True
            except Exception as e:
                emb, err = None, f"{type(e).__name__}: {e}"
            dt = time.time() - t0
            rec = {"label": label, "method": name, "seed": sd, "ok": emb is not None, "seconds": round(dt, 3)}
            if emb is not None:
                fp = embedding_footprint(emb, tgt, label, dt)
                out[label] = {"embedding": emb, "stats": embedding_stats(emb), "seconds": dt, "footprint": fp}
                rec["footprint"] = fp
            else:
                rec["error"] = err
                if skipped:
                    rec["skipped"] = True
            out.attempts.append(rec)

    if verbose:
        print(f"source graph: {n} vars, {src.number_of_edges()} edges | {hardware.summary()}")
        for r in out.attempts:
            if r["ok"]:
                s = r["footprint"]
                print(f"  {r['label']:<22} physical={s['physical']:>5} max_chain={s['max_chain']:>3} "
                      f"mean_chain={s['mean_chain']:.2f} ({r['seconds']:.1f}s)")
            else:
                print(f"  {r['label']:<22} {'SKIPPED' if r.get('skipped') else 'FAILED'}: {r['error']}")
    return out


def select_best(embeddings: dict, key=("physical", "max_chain")):
    """Lowest tuple of the named stats fields wins, e.g. key=("max_chain", "physical"). Returns (label, embedding).
    A cheap structural pick; the real decision can use emulator results instead."""
    if not embeddings:
        raise RuntimeError("no valid embedding available")
    label = min(embeddings, key=lambda k: tuple(embeddings[k]["stats"][f] for f in key))
    return label, embeddings[label]["embedding"]
 

# =========================================================================
# Execution Block: Self-Test
# =========================================================================

if __name__ == "__main__":
    import json
    import sys

    import dimod

    try:
        from solvers.simulated_quantum_annealing.s1_hardware import Hardware
    except ImportError as exc:
        print(f"[-] Could not import Hardware ({exc}). Run from the project root as a module.")
        sys.exit(1)
    try:
        import pandapower.networks as nw
        from solvers.qubo_formulator import QuboFormulator
    except ImportError as exc:
        nw = QuboFormulator = None
        print(f"[!] OPF cases disabled ({exc}); only the synthetic-graph tiers will run.")

    QUICK = False
    TIMEOUT =  20.0 if QUICK else 120.0
    SEEDS = (0,) if QUICK else (0, 1, 2)
    ALL_METHODS = tuple(METHODS)
    OUT_JSON = "embedding_selftest.json"
    RESULTS, CHECKS = [], []          # RESULTS: one entry per (graph, target) cell; CHECKS: (name, passed, detail)

    # ---------------------------------------------------------------- helpers
    def check(name, ok, detail=""):
        CHECKS.append((name, bool(ok), detail))
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    def bqm_from_graph(G):
        G = nx.convert_node_labels_to_integers(G)
        return dimod.BinaryQuadraticModel.from_ising({v: 0.0 for v in G.nodes},
                                                     {(u, v): 1.0 for u, v in G.edges})

    def independent_check(emb, src, tgt):
        """Re-verifies an embedding without trusting is_valid_embedding. Returns a list of problems (empty = fine)."""
        problems = []
        if set(emb) != set(src.nodes):
            problems.append("logical variables do not match the source graph")
        used = {}
        for v, chain in emb.items():
            if not chain:
                problems.append(f"empty chain for {v}")
                continue
            for q in chain:
                if q not in tgt:
                    problems.append(f"qubit {q} not in target")
                if q in used:
                    problems.append(f"qubit {q} shared by {used[q]} and {v}")
                used[q] = v
            if all(q in tgt for q in chain) and not nx.is_connected(tgt.subgraph(chain)):
                problems.append(f"chain of {v} is not connected")
        for u, v in src.edges:
            if u in emb and v in emb and not any(tgt.has_edge(a, b) for a in emb[u] for b in emb[v]):
                problems.append(f"source edge ({u},{v}) has no hardware coupler")
        return problems[:5]

    def run_cell(tier, label, bqm, hw):
        """Runs every method on one (problem, target) pair, prints a table, verifies every success."""
        src = source_graph(bqm)
        n, m = src.number_of_nodes(), src.number_of_edges()
        dens = 100.0 * nx.density(src) if n > 1 else 0.0
        t0 = time.time()
        embs = find_embeddings(bqm, hw, methods=ALL_METHODS, seeds=SEEDS, timeout=TIMEOUT, verbose=False)
        dt = time.time() - t0
        print(f"\n  [{tier}] {label}  ->  {hw.name}   | {n} vars, {m} edges, density {dens:.1f}%   ({dt:.1f}s total)")
        print(f"      {'method':<22}{'status':<8}{'phys':>6}{'max':>5}{'mean':>7}{'cplr':>7}{'secs':>8}   note")

        for r in embs.attempts:
            if r.get("skipped"):
                continue

            secs = r.get("seconds", 0.0)
            if r["ok"]:
                fp = r["footprint"]
                bad = independent_check(embs[r["label"]]["embedding"], src, hw.graph)
                CHECKS.append((f"valid | {label} | {hw.name} | {r['label']}", not bad, "; ".join(bad)))
                note = "verified" if not bad else "INVALID: " + "; ".join(bad)
                print(f"      {r['label']:<22}{'ok':<8}{fp['physical']:>6}{fp['max_chain']:>5}"
                      f"{fp['mean_chain']:>7.2f}{fp['chain_couplers']:>7}{secs:>8.1f}   {note}")
            else:
                print(f"      {r['label']:<22}{('skip' if r.get('skipped') else 'FAIL'):<8}{'':>6}{'':>5}{'':>7}{'':>7}{secs:>8.1f}   "
                      f"{str(r.get('error'))[:90]}")

        entry = {"tier": tier, "label": label, "target": hw.name, "n_vars": n, "n_edges": m,
                 "density_pct": round(dens, 2), "wall_s": round(dt, 2), "chosen": None}
        if embs:
            by_qubits, _ = select_best(embs)
            by_mean, _ = select_best(embs, key=("mean_chain", "max_chain"))
            by_max, _ = select_best(embs, key=("max_chain", "physical"))
            print(f"      best by qubits: {by_qubits} | by mean chain: {by_mean} | by max chain: {by_max}")
            entry["chosen"] = by_qubits
        else:
            print("      -> NO method produced an embedding")
        entry["report"] = embs.report(chosen=entry["chosen"])
        try:
            json.dumps(entry["report"])
        except Exception as exc:
            CHECKS.append((f"report JSON-safe | {label} | {hw.name}", False, str(exc)))
        RESULTS.append(entry)
        return embs

    print("=== Minor-embedding self-test ===")
    print(f"mode={'quick' if QUICK else 'full'} | timeout={TIMEOUT:.0f}s per attempt | seeds={SEEDS} | methods={ALL_METHODS}")

    # ---------------------------------------------------------------- [1] hardware
    print("\n[1] Building hardware targets (small Z4 to provoke failures, full Z12, Z12 with random defects)...")
    z4 = Hardware.ideal_zephyr(m=4, t=4)
    z12 = Hardware.ideal_zephyr(m=12, t=4)
    z12d = z12.with_defects(qubit_fraction=0.02, coupler_fraction=0.005, seed=0)
    for hw in (z4, z12, z12d):
        print("    " + hw.summary())
    caps = {hw.name: max_embeddable_clique(hw.graph) for hw in (z4, z12, z12d)}
    defect_caps = [max_embeddable_clique(z12.with_defects(qubit_fraction=0.02, coupler_fraction=0.005, seed=s).graph)
                   for s in range(5)]
    print(f"    clique capacity over 5 defect seeds (2% qubits, 0.5% couplers): "
          f"min {min(defect_caps)}, max {max(defect_caps)}, all {defect_caps}")
    print("    largest embeddable clique per target:", caps)

    # ---------------------------------------------------------------- [2] API checks
    print("\n[2] API and bookkeeping checks...")
    k12 = bqm_from_graph(nx.complete_graph(12))

    try:
        find_embeddings(k12, z12, methods=("nope",), verbose=False)
        check("unknown method raises ValueError", False)
    except ValueError:
        check("unknown method raises ValueError", True)

    e1 = find_embeddings(k12, z12, methods=("clique", "minorminer", "mm_from_clique"), seeds=(0,),
                         timeout=TIMEOUT, verbose=False, skip_minorminer_if_clique_found=True)
    if "clique" in e1:
        check("skip flag: only clique attempted once clique succeeds",
              {r["label"] for r in e1.attempts} == {"clique"}, str([r["label"] for r in e1.attempts]))
    else:
        check("K12 clique embedding on Z12 succeeds", False, str(e1.attempts))

    e2 = find_embeddings(k12, z12, methods=ALL_METHODS, seeds=(0,), timeout=TIMEOUT, verbose=False)
    check("EmbeddingSet behaves as a dict", isinstance(e2, dict))
    check("every method/seed attempted and recorded", len(e2.attempts) == 4, f"{len(e2.attempts)} attempts")
    check("successes carry stats, footprint and seconds",
          all({"embedding", "stats", "seconds", "footprint"} <= set(v) for v in e2.values()))
    try:
        json.dumps(e2.report(chosen=None))
        check("report() is JSON-safe", True)
    except Exception as exc:
        check("report() is JSON-safe", False, str(exc))
    if e2:
        lab, emb = select_best(e2)
        fp = embedding_footprint(emb, z12.graph, lab, e2[lab]["seconds"])
        check("footprint with target has chain_couplers and device usage", {"chain_couplers", "device_qubits_used_pct"} <= set(fp))
        try:
            json.dumps(fp)
            check("footprint is JSON-safe", True)
        except Exception as exc:
            check("footprint is JSON-safe", False, str(exc))

    try:
        select_best(EmbeddingSet())
        check("select_best on an empty set raises RuntimeError", False)
    except RuntimeError:
        check("select_best on an empty set raises RuntimeError", True)

    e4 = find_embeddings(k12, z12, methods=("mm_from_clique",), seeds=(0,), timeout=TIMEOUT, verbose=False)
    check("mm_from_clique without clique is logged as skipped",
          len(e4) == 0 and e4.attempts and e4.attempts[0].get("skipped") and "needs a successful" in str(e4.attempts[0].get("error")))

    if caps[z4.name]:
        big = bqm_from_graph(nx.complete_graph(caps[z4.name] + 5))
        e3 = find_embeddings(big, z4, methods=ALL_METHODS, seeds=(0,), timeout=TIMEOUT, verbose=False)
        check("precheck: impossible dense source returns empty set at once",
              len(e3) == 0 and e3.attempts and e3.attempts[0]["method"] == "precheck")
    else:
        print("    [skip] precheck test: busclique reports no clique capacity for Z4")

    huge = bqm_from_graph(nx.complete_graph(101))    # 5050 edges > 5032 couplers on Z4
    e5 = find_embeddings(huge, z4, methods=ALL_METHODS, seeds=(0,), timeout=TIMEOUT, verbose=False)
    check("edge-count precheck reports a PROOF", len(e5) == 0 and e5.attempts and e5.attempts[0].get("proven") is True)

    cap12 = caps[z12.name]
    if cap12:
        e6 = find_embeddings(bqm_from_graph(nx.complete_graph(cap12)), z12, methods=("clique",), verbose=False)
        check(f"stated clique capacity is real: K{cap12} embeds on {z12.name}", len(e6) == 1)
        if not QUICK:
            for extra in (1, 3, 6):
                k = cap12 + extra
                pr = find_embeddings(bqm_from_graph(nx.complete_graph(k)), z12, methods=("minorminer",), seeds=(0,),
                                     timeout=TIMEOUT, verbose=False, skip_beyond_clique_capacity=False)
                print(f"    probe: K{k} (capacity+{extra}) minorminer -> "
                      f"{'EMBEDDED' if pr else 'no embedding'} ({pr.attempts[0]['seconds']:.0f}s)")

    # ---------------------------------------------------------------- [3] tier A: synthetic graphs
    print("\n[3] Tier A: synthetic source graphs, dense to sparse, on Z4 (expect failures) and Z12")
    complete_sizes = (8, 24, 64) if QUICK else (8, 24, 48, 96, 144, 200)
    sparse = [("grid 8x8", nx.grid_2d_graph(8, 8)), ("ER n=100 p=0.1", nx.gnp_random_graph(100, 0.1, seed=1))]
    if not QUICK:
        sparse += [("3-regular n=100", nx.random_regular_graph(3, 100, seed=1)),
                   ("Barabasi-Albert n=150 m=3", nx.barabasi_albert_graph(150, 3, seed=1))]
    for hw in (z4, z12):
        graphs = [(f"K{n} complete", nx.complete_graph(n)) for n in complete_sizes] + sparse
        if caps[hw.name]:
            graphs.append((f"K{caps[hw.name] + 5} (capacity+5)", nx.complete_graph(caps[hw.name] + 5)))
        for label, G in graphs:
            run_cell("A", label, bqm_from_graph(G), hw)

    # ---------------------------------------------------------------- [4] tier B: OPF QUBOs
    if QuboFormulator is not None:
        print("\n[4] Tier B: DC-OPF QUBOs of increasing size on Z12 (largest also on Z12 with defects)")
        opf_cases = [("case4gs dc_ptdf", "case4gs", {}), ("case9 dc_ptdf", "case9", {}),
                     ("case14 dc_ptdf", "case14", {}), ("case30 dc_ptdf", "case30", {}),
                     ("case118 dc_ptdf 10MW", "case118", {})]
        if not QUICK:
            opf_cases[4:4] = [("case57 dc_ptdf", "case57", {})]
            opf_cases += [("case118 dc_ptdf 5MW", "case118", dict(mw_precision=5.0)),
                          ("case30 dc_theta", "case30", dict(formulation="dc_theta")),
                          ("case118 dc_theta", "case118", dict(formulation="dc_theta")),
                          ("case300 dc_ptdf 10MW", "case300", {})]
        for label, fname, kw in opf_cases:
            print(f"\n  -- formulating {label}")
            try:
                net = getattr(nw, fname)()
                if not net.ext_grid.empty:
                    net.ext_grid["min_p_mw"] = 0.0
                tf = time.time()
                f = QuboFormulator(**{**dict(formulation="dc_ptdf", mw_precision=10.0), **kw})
                bqm, cx = f._formulate_qubo(net)
                print(f"     formulated in {time.time() - tf:.1f}s: {len(bqm.variables)} logical qubits, "
                      f"{bqm.num_interactions} couplings, density {cx['quantum_domain_qubo']['density_percent']}%")
            except Exception as exc:
                print(f"     FORMULATION FAILED: {type(exc).__name__}: {exc}")
                RESULTS.append({"tier": "B", "label": label, "target": None, "formulation_error": str(exc),
                                "report": {"attempts": []}, "chosen": None})
                continue
            if not bqm.variables:
                print("     0 variables, skipped")
                continue
            targets = (z12, z12d) if "118" in label or "300" in label else (z12,)
            for hw in targets:
                run_cell("B", label, bqm, hw)
    else:
        print("\n[4] Tier B skipped (formulator or pandapower not importable).")

    # ---------------------------------------------------------------- [5] summary
    print("\n" + "=" * 90 + "\nSUMMARY\n" + "=" * 90)
    per_method, reasons, wins = {}, Counter(), Counter()
    for e in RESULTS:
        for r in e["report"]["attempts"]:
            d = per_method.setdefault(r["method"], [0, 0])
            d[1] += 1
            d[0] += int(r["ok"])
            if not r["ok"]:
                reasons[str(r.get("error"))[:70]] += 1
        if e.get("chosen"):
            wins[e["chosen"].split("_seed")[0]] += 1
    print("success rate per method (successful attempts / all attempts):")
    for m, (ok, tot) in per_method.items():
        print(f"  {m:<16}{ok:>4}/{tot:<4} ({100.0 * ok / max(1, tot):.0f}%)   chosen as best by qubit count in {wins[m]} cells")
    unembedded = [f"{e['label']} on {e['target']}" for e in RESULTS if not e.get("chosen") and e.get("target")]
    print(f"\ncells where NO method produced an embedding: {len(unembedded)}")
    for u in unembedded:
        print("  - " + u)
    if reasons:
        print("\nmost common failure reasons:")
        for msg, cnt in reasons.most_common(6):
            print(f"  {cnt:>4}x  {msg}")

    failed = [c for c in CHECKS if not c[1]]
    print(f"\nchecks: {len(CHECKS) - len(failed)} passed, {len(failed)} failed (includes independent validity of every embedding)")
    for name, _, detail in failed:
        print(f"  FAILED: {name}  {detail}")

    with open(OUT_JSON, "w") as fh:
        json.dump({"settings": {"quick": QUICK, "timeout": TIMEOUT, "seeds": SEEDS, "methods": ALL_METHODS,
                                "clique_capacity": caps},
                   "results": RESULTS, "checks": CHECKS}, fh, default=str, indent=1)
    print(f"\nfull results written to {OUT_JSON}")
    sys.exit(1 if failed else 0)