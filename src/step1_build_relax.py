"""
STEP 1: build a Mo-doped Ni(111) surface, adsorb ethane, relax to the 0 K
ground state, and compute the adsorption energy.

Writes outputs/relaxed.traj, which steps 2 and 3 read.
"""

import sys
from pathlib import Path

import numpy as np
from ase.build import add_adsorbate, fcc111, molecule
from ase.constraints import FixAtoms
from ase.optimize import BFGS

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def new_calculator():
    """A fresh calculator for each system.

    One calculator object cannot be reused across systems with different
    atom counts - it caches a results buffer sized to the first structure
    it sees and then raises a numpy broadcast error.
    """
    unit = load_predict_unit(path=str(config.MODEL_PATH), device=config.DEVICE)
    return FAIRChemCalculator(unit, task_name=config.TASK_NAME)


def make_clean_slab():
    """Ni(111) slab with 3 Mo atoms substituted into the top layer."""
    s = fcc111("Ni", size=config.SLAB_SIZE, vacuum=config.VACUUM)

    # fcc111 tags every atom with its layer number: 1 = top, 4 = bottom.
    layer = s.get_tags()
    top = [i for i in range(len(s)) if layer[i] == 1]
    bottom = [i for i in range(len(s))
              if layer[i] > config.SLAB_SIZE[2] - config.N_FIXED_LAYERS]

    # Spread the dopants out so they act as separate active sites.
    for i in (top[0], top[len(top) // 2], top[-1]):
        s[i].symbol = "Mo"

    # Bulk metal underneath would hold these atoms still; we mimic that.
    s.set_constraint(FixAtoms(indices=bottom))
    s.pbc = True
    return s, top


def main():
    config.check_checkpoint()

    # --- bare surface ------------------------------------------------------
    clean, top_layer = make_clean_slab()
    clean.set_tags([1 if i in top_layer else 0 for i in range(len(clean))])
    clean.calc = new_calculator()
    BFGS(clean).run(fmax=config.FMAX, steps=300)
    E_surface = clean.get_potential_energy()
    print(f"bare surface     = {E_surface:9.3f} eV")

    # --- surface + ethane --------------------------------------------------
    slab, top_layer = make_clean_slab()
    n_metal = len(slab)

    ads = molecule(config.ADSORBATE)
    ads.rotate(90, "x")                    # lay the C-C bond flat
    ads.translate([0.0, 0.0, -ads.positions[:, 2].min()])

    mo_x, mo_y = slab.positions[top_layer[0], :2]
    add_adsorbate(slab, ads, height=config.ADS_HEIGHT,
                  position=(float(mo_x), float(mo_y)))
    slab.center(vacuum=config.VACUUM, axis=2)   # restore vacuum above ethane

    ads_indices = list(range(n_metal, len(slab)))

    # OC20 tag convention: 0 = bulk, 1 = surface, 2 = adsorbate.
    # This is NOT the layer numbering fcc111 gave us - overwrite it, or the
    # model sees a system with no adsorbate at all.
    tags = np.zeros(len(slab), dtype=int)
    tags[top_layer] = 1
    tags[ads_indices] = 2
    slab.set_tags(tags)

    slab.calc = new_calculator()
    start_force = np.linalg.norm(
        slab.get_forces(apply_constraint=False), axis=1).max()
    print(f"largest starting force = {start_force:.2f} eV/A  (over ~10 = trouble)")

    BFGS(slab, trajectory=str(config.RELAXED_TRAJ)).run(
        fmax=config.FMAX, steps=300)
    E_together = slab.get_potential_energy()
    print(f"surface + ethane = {E_together:9.3f} eV")

    # --- free molecule -----------------------------------------------------
    gas = molecule(config.ADSORBATE)
    gas.set_cell([15, 15, 15])
    gas.center()
    gas.pbc = True
    gas.set_tags([2] * len(gas))
    gas.calc = new_calculator()
    BFGS(gas).run(fmax=config.FMAX, steps=200)
    E_gas = gas.get_potential_energy()
    print(f"free ethane      = {E_gas:9.3f} eV")

    # --- adsorption energy -------------------------------------------------
    # A thermodynamic cycle, same logic as Hess's law. Absolute energies from
    # the model are arbitrary; only differences carry meaning.
    E_ads = E_together - E_surface - E_gas
    print(f"\nE_ads = {E_ads:.4f} eV")
    if E_ads < -0.5:
        print("strongly bound (chemisorbed)")
    elif E_ads < 0:
        print("weakly bound (physisorbed)")
    else:
        print("not bound - ethane prefers the gas phase")

    return E_ads


if __name__ == "__main__":
    main()
