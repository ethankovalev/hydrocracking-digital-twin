"""
STEP 1: build a Mo-doped Ni(111) surface, adsorb ethane, relax to the 0 K
ground state, and compute the adsorption energy.

Writes outputs/relaxed.traj, which steps 2 and 3 read.

The OC20 task is not trained on isolated molecules, so asking it for free ethane gives an
unreliable number. Instead we relax a second slab+ethane system with the
molecule parked far above the surface. The model still sees a slab, and the
slab energy cancels in E_together - E_far.
"""

import sys
from pathlib import Path
from ase.build import add_adsorbate, fcc111, molecule
from ase.constraints import FixAtoms
from ase.optimize import BFGS

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

SEPARATION = 8.0        # height used for the non-interacting reference


def new_calculator():
    """A fresh calculator for each system.

    One calculator object cannot be reused across systems with different
    atom counts - it caches a results buffer sized to the first structure
    it sees and then raises a numpy broadcast error.
    """
    unit = load_predict_unit(path=str(config.MODEL_PATH), device=config.DEVICE)
    return FAIRChemCalculator(unit, task_name=config.TASK_NAME)


def make_clean_slab():
    """Ni(111) slab with 3 Mo atoms in the top layer, ready for the model."""
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

    # OC20 tags: 1 = surface, 0 = bulk (adsorbate atoms get 2 in build).
    # This replaces the layer numbering, which the model would misread.
    s.set_tags([1 if i in top else 0 for i in range(len(s))])

    # Every system here shares one cell height so the energies compare.
    cell = s.get_cell()
    cell[2, 2] = config.CELL_Z
    s.set_cell(cell)
    s.center(axis=2)
    return s, top


def build(height):
    """Mo-doped slab with ethane `height` above the surface.

    height = config.ADS_HEIGHT -> the adsorbed system
    height = SEPARATION        -> the non-interacting reference

    Both systems come from here, so they differ only in that one number.
    """
    slab, top = make_clean_slab()
    n_metal = len(slab)

    ads = molecule(config.ADSORBATE)
    ads.rotate(90, "x")                       # lay the C-C bond flat

    x = slab.positions[top[0], 0]             # top[0] is one of the Mo atoms
    y = slab.positions[top[0], 1]
    add_adsorbate(slab, ads, height=height, position=(float(x), float(y)))

    # add_adsorbate measures `height` to atom 0, which is a carbon, so the
    # hydrogens end up about 1 A closer than asked. Shift the molecule so
    # its lowest atom sits exactly `height` above the highest metal atom.
    # This has to happen after placement: a shift applied to the molecule
    # beforehand is simply undone by add_adsorbate.
    z_metal = max(slab.positions[i, 2] for i in range(n_metal))
    z_low = min(slab.positions[i, 2] for i in range(n_metal, len(slab)))
    shift = (z_metal + height) - z_low
    for i in range(n_metal, len(slab)):
        slab.positions[i, 2] += shift

    slab.center(axis=2)

    ads_indices = list(range(n_metal, len(slab)))
    tags = list(slab.get_tags())
    for i in ads_indices:
        tags[i] = 2
    slab.set_tags(tags)
    return slab, ads_indices


def relax(atoms, label, traj=None):
    """Relax to fmax, and refuse to return an energy from a non-minimum."""
    if not BFGS(atoms, trajectory=traj).run(fmax=config.FMAX, steps=300):
        raise RuntimeError(f"{label} did not converge in 300 steps")
    return atoms.get_potential_energy()


def gap(atoms, ads_indices):
    """Shortest distance from any adsorbate atom to any metal atom."""
    metal = [i for i in range(len(atoms)) if i not in ads_indices]
    return min(atoms.get_distances(a, metal, mic=True).min()
               for a in ads_indices)


def main():
    config.check_checkpoint()

    # 1. bare surface
    clean, _ = make_clean_slab()
    clean.calc = new_calculator()
    E_surface = relax(clean, "bare surface")

    # 2. ethane adsorbed
    slab, ads = build(config.ADS_HEIGHT)
    slab.calc = new_calculator()
    print(f"closest contact at start = {gap(slab, ads):.2f} Å  (want > 2.0)")
    E_together = relax(slab, "surface + ethane", traj=str(config.RELAXED_TRAJ))

    # 3. reference: same system, ethane parked out of reach.
    far, far_ads = build(SEPARATION)
    far.calc = new_calculator()
    far_start = gap(far, far_ads)
    print(f"reference contact at start = {far_start:.2f} Å  "
          f"(should be about {SEPARATION:.1f})")
    if far_start < SEPARATION - 1.0:
        raise RuntimeError(
            f"reference built at only {far_start:.2f} Å - wrong height was "
            "passed to build(), so this is not a valid reference")

    E_far = relax(far, "far reference")
    if gap(far, far_ads) < SEPARATION - 1.0:
        raise RuntimeError("the reference ethane drifted onto the slab")

    E_ads = E_together - E_far

    carbons = [i for i in ads if slab[i].symbol == "C"]
    cc = slab.get_distance(carbons[0], carbons[1], mic=True)

    print(f"\nbare surface          = {E_surface:9.3f} eV")
    print(f"surface + ethane      = {E_together:9.3f} eV")
    print(f"slab + distant ethane = {E_far:9.3f} eV")
    print(f"implied E_gas         = {E_far - E_surface:9.3f} eV")
    print(f"C-C after relaxation  = {cc:.2f} Å   (expect ~1.53)")
    print(f"closest contact       = {gap(slab, ads):.2f} Å")
    print(f"\nE_ads = {E_ads:.4f} eV")

    if cc > 2.0:
        print("WARNING: the C-C bond broke during relaxation")
    if E_ads < -0.5:
        print("strongly bound (chemisorbed)")
    elif E_ads < 0:
        print("weakly bound (physisorbed)")
    else:
        print("not bound - ethane prefers the gas phase")

    return E_ads


if __name__ == "__main__":
    main()