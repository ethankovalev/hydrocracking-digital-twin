"""
STEP 3: nudged elastic band search for the C-C scission activation energy.

Relaxation finds valley floors. This finds the mountain pass between two
valleys - the transition state - which is what an activation energy is.

NOTE: fairchem ships CatTSunami (Wander, Shuaibi, Kitchin, Ulissi, Zitnick),
purpose-built for transition states with pretrained GNNs. Plain ASE NEB is
used here so the mechanics are visible; CatTSunami is the production route.
"""

import sys
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import BFGS

try:
    from ase.mep import NEB, NEBTools          # ASE >= 3.23
except ImportError:
    from ase.neb import NEB, NEBTools          # older ASE

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def main():
    config.check_checkpoint()
    if not config.RELAXED_TRAJ.exists():
        raise FileNotFoundError("Run step 1 first - outputs/relaxed.traj is missing.")

    # Every NEB image has the same atom count, so one predict unit is safe.
    # ASE still needs a separate calculator wrapper per image.
    unit = load_predict_unit(path=str(config.MODEL_PATH), device=config.DEVICE)

    def new_calculator():
        return FAIRChemCalculator(unit, task_name=config.TASK_NAME)

    # --- initial state -----------------------------------------------------
    initial = read(str(config.RELAXED_TRAJ))
    initial.calc = new_calculator()

    tags = initial.get_tags()
    carbons = [i for i in range(len(initial))
               if tags[i] == 2 and initial[i].symbol == "C"]
    hydrogens = [i for i in range(len(initial))
                 if tags[i] == 2 and initial[i].symbol == "H"]

    print(f"carbons {carbons}, {len(hydrogens)} hydrogens")
    print(f"starting C-C = {initial.get_distance(*carbons, mic=True):.3f} A")

    # --- final state -------------------------------------------------------
    # Build by MOVING atoms in a copy. NEB requires identical atom ordering
    # between endpoints; rebuilding from scratch could reorder them and NEB
    # would silently interpolate nonsense.
    final = initial.copy()

    # Assign each hydrogen to its nearest carbon.
    group = {carbons[0]: [carbons[0]], carbons[1]: [carbons[1]]}
    for h in hydrogens:
        d0 = initial.get_distance(h, carbons[0], mic=True)
        d1 = initial.get_distance(h, carbons[1], mic=True)
        group[carbons[0] if d0 < d1 else carbons[1]].append(h)

    # Pull apart along the C-C axis, flattened into the surface plane so the
    # fragments slide across the metal rather than lifting off it.
    axis = initial.positions[carbons[1]] - initial.positions[carbons[0]]
    axis[2] = 0.0
    axis /= np.linalg.norm(axis)

    shift = (config.SEPARATION - initial.get_distance(*carbons, mic=True)) / 2.0
    final.positions[group[carbons[0]]] -= axis * shift
    final.positions[group[carbons[1]]] += axis * shift

    final.calc = new_calculator()
    print("\nrelaxing the split-fragment final state...")
    BFGS(final, logfile="-").run(fmax=config.FMAX, steps=300)
    print(f"final C-C = {final.get_distance(*carbons, mic=True):.3f} A")

    # --- reaction path -----------------------------------------------------
    images = [initial]
    for _ in range(config.N_IMAGES):
        img = initial.copy()
        img.calc = new_calculator()
        images.append(img)
    images.append(final)

    # climb=True: the highest image has its spring force removed and its
    # force along the path inverted, driving it UP onto the saddle point.
    # Without it, no image sits on the peak and Ea comes out too low.
    neb = NEB(images, climb=True)

    # IDPP interpolates in interatomic distances rather than straight lines,
    # so atoms are not pushed through each other on the way.
    neb.interpolate(method="improvedtangent")

    print(f"\nrelaxing the {len(images)}-image path...")
    BFGS(neb, trajectory=str(config.NEB_TRAJ), logfile="-").run(
        fmax=config.FMAX, steps=200)

    # --- barrier -----------------------------------------------------------
    Ea_forward, delta_E = NEBTools(images).get_barrier()
    energies = np.array([img.get_potential_energy() for img in images])
    relative = energies - energies[0]

    print("\n" + "=" * 58)
    print("  energy along the path (eV, relative to start):")
    for i, e in enumerate(relative):
        print(f"    image {i}: {e:7.3f}  {'#' * max(0, int(e * 20))}")
    print("-" * 58)
    print(f"  activation energy Ea = {Ea_forward:.3f} eV")
    print(f"  reaction energy dE   = {delta_E:.3f} eV")
    print("=" * 58)

    if Ea_forward < 0.1:
        print("\n  Ea suspiciously small - check the final state actually")
        print("  differs from the initial one and did not relax back.")

    write(str(config.OUTPUT_DIR / "neb_path.xyz"), images)
    np.save(str(config.EA_FILE), np.array([Ea_forward]))
    return Ea_forward


if __name__ == "__main__":
    main()
