import warnings
import logging
import os
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger().setLevel(logging.ERROR)

import itertools
from tqdm import tqdm
from scripts.opf_annealing_benchmark import run_pipeline

# Import your grid generators
import pandapower.networks as pn
from homemade_grids.small_grids import case3_low_gen, case4_high_gen

if __name__ == "__main__":
    
    # 1. Define the Grids
    grid_generators = [
        # --- Micro / Fundamental Verification ---
        #case3_low_gen,
        #case4_high_gen,
        #pn.case5,          # 5 buses
        #pn.case6ww,        # 6-bus standard case
        
        # --- Small / Feasible Hardware Bounds ---
        #pn.case9,          # 9 buses
        #pn.case14,         # 14 buses
        #pn.case24_ieee_rts,# 24-bus IEEE Reliability Test System
        #pn.case30,         # 30 buses
        #pn.case39,         # 39 buses (New England)
        #pn.case57,         # 57 buses
        
        # --- Medium / The Topological Cliff (100 - 300 buses) ---
        pn.case89pegase,   # 89 buses (European Pegase subset)
        pn.case118,        # 118 buses
        #pn.case145,        # 145 buses
        #pn.case300,        # 300 buses
        
        # --- Large / Unfeasible Mathematical Baselines (1000+ buses) ---
        # These will instantly fail embedding on current quantum hardware, 
        # but are excellent for testing the pure logical solver (SA) scaling.
        # pn.case1354pegase, # 1354 buses (European grid)
        # pn.case2869pegase, # 2869 buses
        # pn.case3120sp,     # 3120 buses (Polish grid)
        # pn.case6470rte     # 6470 buses (French grid)
    ]
    
    # 2. Define the parameter space
    formulations = ["dc_ptdf"] # , "dc_theta"
    encodings = ["radix"] # , "unary"
    precisions = [50.0, 20.0, 10.0]
    
    # 3. Define Tiered SQA Hyperparameters based on grid size (number of buses)
    # The larger the search space, the slower the temperature schedule (sweeps) must be.
    def get_hyperparameters(net):
        n_buses = len(net.bus)
        if n_buses <= 6:
            # Micro grids: Search space is tiny, quick convergence
            return {"num_reads": 1000, "num_sweeps": 5000}
        elif n_buses <= 57:
            # Small grids: Moderate search space
            return {"num_reads": 2000, "num_sweeps": 20000}
        elif n_buses <= 300:
            # Medium grids: Massive search space, highly susceptible to local minima
            return {"num_reads": 5000, "num_sweeps": 50000}
        else:
            # Large grids (1000+): Requires extreme classical compute time
            return {"num_reads": 10000, "num_sweeps": 100000}
    
    total_runs = len(grid_generators) * len(formulations) * len(encodings) * len(precisions)

    print("\n" + "="*80)
    print(f" INITIATING BATCH SWEEP: {total_runs} COMBINATIONS ".center(80, "="))
    print("="*80 + "\n")

    # Initialize tqdm progress bar
    with tqdm(total=total_runs, desc="Sweeping", unit="run", dynamic_ncols=True) as pbar:
        
        # 4. Outer loop for Grids
        for grid_func in grid_generators:
            
            # Generate a fresh copy of the grid to evaluate its size for hyperparameter assignment
            net = grid_func()
            grid_name = net.name if hasattr(net, 'name') and net.name else 'Unknown'
            
            # Fetch dynamic hyperparameters based on grid size
            hyperparams = get_hyperparameters(net)
            current_reads = hyperparams["num_reads"]
            current_sweeps = hyperparams["num_sweeps"]
            
            # 5. Inner loop for configurations
            for form, enc, prec in itertools.product(formulations, encodings, precisions):
                
                # Regenerate the grid to ensure state isolation between runs
                net = grid_func()
                
                # Enforce standard rule: prevent external grid from exporting power to align with PQ formulations
                if hasattr(net, 'ext_grid') and not net.ext_grid.empty:
                    net.ext_grid['min_p_mw'] = 0.0
                
                # Update progress bar info text dynamically
                pbar.set_postfix(grid=grid_name[:10], form=form[-4:], enc=enc[:3], prec=prec)
                
                config = {
                    "formulation": form,
                    "encoding": enc,
                    "mw_precision": prec,
                    "penalty_safety": 1.1,
                    "ptdf_threshold": 0.05,
                }
                
                try:
                    # Pass the dynamic hyperparameters to the pipeline
                    run_pipeline(config, net, current_reads, current_sweeps)
                except Exception as e:
                    # Use tqdm.write so error messages don't break the progress bar UI
                    tqdm.write(f"\n[!] RUN FAILED: Grid={grid_name}, Form={form}, Enc={enc}, Prec={prec}")
                    tqdm.write(f"    Error: {str(e)}")
                    
                # Advance the progress bar by 1
                pbar.update(1)
                
    print("\n" + "="*80)
    print(" BATCH SWEEP COMPLETE. ALL DATA SAVED TO JSON. ".center(80, "="))
    print("="*80 + "\n")