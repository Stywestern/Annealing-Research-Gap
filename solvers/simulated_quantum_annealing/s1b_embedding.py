"""Step 1b: minor embedding of the logical problem graph into the Zephyr graph.
Current methods are: 
    Minorminer (2014, A practical heuristic for finding graph minors, papers version)
    Busclique
    
"""
from __future__ import annotations
 
import time
from collections import Counter
 
import networkx as nx

 #########################################################################################################
def source_graph(bqm) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(bqm.variables)
    g.add_edges_from((u, v) for u, v, _ in bqm.iter_quadratic())
    return g
 
 
def embedding_stats(embedding: dict) -> dict:
    lens = [len(c) for c in embedding.values()]
    if not lens:
        return {"logical": 0, "physical": 0, "max_chain": 0, "mean_chain": 0.0, "hist": {}}
    return {"logical": len(lens), "physical": sum(lens), "max_chain": max(lens),
            "mean_chain": sum(lens) / len(lens), "hist": dict(sorted(Counter(lens).items()))}
 
 
def _is_valid(embedding, src: nx.Graph, tgt: nx.Graph) -> bool:
    from dwave.embedding import is_valid_embedding
    return bool(is_valid_embedding(embedding, src, tgt))
 
 
def embed_minorminer(src: nx.Graph, tgt: nx.Graph, seed: int = 0, tries: int = 20, timeout: float = 120.0):
    import minorminer
    emb = minorminer.find_embedding(src, tgt, random_seed=seed, tries=tries, timeout=timeout)
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
 
 
def find_embeddings(bqm, hardware, methods=("clique", "minorminer"), seeds=(0, 1, 2),
                    timeout: float = 36000.0, verbose: bool = True, density_skip_minorminer: float = 0.9,
                    skip_minorminer_if_clique_found: bool = False) -> dict:
    """Returns {label: {"embedding", "stats", "seconds"}} for every method/seed that succeeded.
 
    density_skip_minorminer : if the source graph's density is at or above this (e.g. your balance
        clique at ~100%), a capacity precheck runs FIRST via max_embeddable_clique -- if the source is
        provably too large to ever embed, minorminer is skipped entirely (no amount of search time
        fixes a hard capacity shortfall) and this returns {} immediately rather than after a multi-
        minute timeout per seed. Set to None to disable this precheck.
    skip_minorminer_if_clique_found : if True and "clique" succeeds, minorminer is not attempted at
        all. False (default) keeps both, e.g. for a benchmarking comparison between the two methods;
        set True once you've decided busclique's clique embedding is simply the right tool for a
        complete/near-complete source graph and don't want to pay minorminer's search cost for nothing.
    """
    src, tgt = source_graph(bqm), hardware.graph
    out = {}
 
    if (density_skip_minorminer is not None and src.number_of_nodes() > 1
            and nx.density(src) >= density_skip_minorminer):
        cap = max_embeddable_clique(tgt)
        if cap and src.number_of_nodes() > cap:
            if verbose:
                print(f"source graph: {src.number_of_nodes()} vars, density {nx.density(src):.2%} "
                      f"| {hardware.summary()}")
                print(f"  [precheck] source needs a clique of {src.number_of_nodes()} nodes; this "
                      f"target can embed at most {cap} -- NO embedding is possible here, skipping "
                      f"minorminer entirely (it cannot succeed where busclique's own clique capacity "
                      f"says there isn't room)")
            return out
 
    if "clique" in methods:
        t0 = time.time()
        emb = embed_clique(src, tgt, verbose=verbose)
        if emb and _is_valid(emb, src, tgt):
            out["clique"] = {"embedding": emb, "stats": embedding_stats(emb), "seconds": time.time() - t0}
 
    run_minorminer = "minorminer" in methods and not (skip_minorminer_if_clique_found and "clique" in out)
    if run_minorminer:
        for sd in seeds:
            t0 = time.time()
            emb = embed_minorminer(src, tgt, seed=sd, timeout=timeout)
            if emb and _is_valid(emb, src, tgt):
                out[f"minorminer_seed{sd}"] = {"embedding": emb, "stats": embedding_stats(emb),
                                               "seconds": time.time() - t0}
 
    if verbose:
        print(f"source graph: {src.number_of_nodes()} vars, {src.number_of_edges()} edges | {hardware.summary()}")
        if not out:
            print("  no embedding found")
        for k, v in out.items():
            s = v["stats"]
            print(f"  {k:<20} physical={s['physical']:>5} max_chain={s['max_chain']:>3} "
                  f"mean_chain={s['mean_chain']:.2f} ({v['seconds']:.1f}s)")
    return out
 
 
def select_best(embeddings: dict):
    """Fewest physical qubits, ties broken by shorter longest chain. Returns (label, embedding)."""
    if not embeddings:
        raise RuntimeError("no valid embedding available")
    label = min(embeddings, key=lambda k: (embeddings[k]["stats"]["physical"],
                                           embeddings[k]["stats"]["max_chain"]))
    return label, embeddings[label]["embedding"]
 

# =========================================================================
# Execution Block: Self-Test
# =========================================================================
if __name__ == "__main__":
    import pandapower.networks as nw
    import time
    
    # We must explicitly import QuboFormulator and Hardware. 
    # Adjust the import paths based on your actual directory structure.
    try:
        from solvers.qubo_formulator import QuboFormulator
        from solvers.simulated_quantum_annealing.s1_hardware import Hardware
        # Assuming find_embeddings and select_best are available in your namespace
    except ImportError:
        print("[-] Could not import required modules. Ensure this script is run from the project root.")
        exit(1)

    print("=== Testing Minor-Embedding Limits: Case118 ===")
    
    # 1. Generate a standardized grid
    print("[1] Loading IEEE 118-bus network (case118)...")
    net = nw.case118()
    if not net.ext_grid.empty:
        net.ext_grid['min_p_mw'] = 0.0

    # 2. Formulate the Logical BQM
    print("[2] Formulating DC-PTDF QUBO (Radix, 10 MW precision)...")
    formulator = QuboFormulator(formulation="dc_ptdf", mw_precision=10.0)
    bqm, complexity = formulator._formulate_qubo(net)
    
    logical_vars = len(bqm.variables)
    logical_edges = len(bqm.quadratic)
    density = (logical_edges / (logical_vars * (logical_vars - 1) / 2)) * 100 if logical_vars > 1 else 0
    print(f"    -> Logical Problem Size: {logical_vars} variables, {logical_edges} edges.")
    print(f"    -> Graph Density: {density:.1f}%")
    
    if logical_vars == 0:
        print("    -> ERROR: 0 variables.")
        exit(1)

    # 3. Define the Hardware Constraint
    # Using Z15 (15 * 15 * 24 = 5400 qubits), the full scale D-Wave Advantage2
    print("\n[3] Initializing Target Hardware (Full-Scale Zephyr Z15, ~5400 qubits)...")
    target_hardware = Hardware.ideal_zephyr(m=15, t=4)

    # 4. Execute the Embedding Sweep exactly as you wrote it
    print("\n[4] Running Multi-Heuristic Minor-Embedding Sweep...")
    print("    [!] WARNING: Attempting to embed a highly dense 240+ node graph.")
    
    t0 = time.time()
    # We increase the timeout to 60s so minorminer doesn't fail just because it's slow.
    results = find_embeddings(
        bqm=bqm, 
        hardware=target_hardware, 
        methods=("clique", "minorminer"), 
        seeds=(0, 1, 2), 
        timeout=60.0, 
        verbose=True
    )
    t1 = time.time()

    # 5. Select the Winner
    if results:
        print(f"\n[5] Embedding SUCCESSFUL! (Time taken: {t1-t0:.2f} seconds)")
        best_label, best_emb = select_best(results)
        best_stats = results[best_label]["stats"]
        
        print(f"    -> WINNER: {best_label}")
        print(f"    -> Physical Footprint: {best_stats['physical']} qubits.")
        print(f"    -> Longest Chain: {best_stats['max_chain']} qubits.")
        
        overhead = (best_stats['physical'] / logical_vars) if logical_vars > 0 else 0
        print(f"    -> Qubit Overhead Factor: {overhead:.1f}x physical-to-logical ratio.")
    else:
        print(f"\n[-] FATAL: All embedding attempts FAILED. (Time taken: {t1-t0:.2f} seconds)")