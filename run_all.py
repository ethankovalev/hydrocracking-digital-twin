"""
Run the pipeline.

    python run_all.py            # every step in order
    python run_all.py 0 4        # only steps 0 and 4
    python run_all.py 0          # just clean the telemetry

Steps 1-3 call a graph neural network thousands of times and want a CUDA
GPU. On a laptop CPU they will run, but slowly.
"""

import sys

from src import (step0_clean_scada, step1_build_relax, step2_langevin_md,
                 step3_neb_barrier, step4_pinn)

STEPS = {
    0: ("clean SCADA telemetry", step0_clean_scada.main),
    1: ("build + relax (0 K)", step1_build_relax.main),
    2: ("Langevin MD (675 K)", step2_langevin_md.main),
    3: ("NEB activation energy", step3_neb_barrier.main),
    4: ("physics-informed NN", step4_pinn.main),
}

if __name__ == "__main__":
    try:
        wanted = [int(a) for a in sys.argv[1:]] or sorted(STEPS)
    except ValueError:
        print("Usage: python run_all.py [step numbers, e.g. 0 1 4]")
        sys.exit(1)

    for n in wanted:
        if n not in STEPS:
            print(f"No step {n}. Valid steps: {sorted(STEPS)}")
            sys.exit(1)
        name, fn = STEPS[n]
        print(f"\n{'=' * 62}\n  STEP {n}: {name}\n{'=' * 62}")
        fn()
