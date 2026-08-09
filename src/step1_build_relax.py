"""
STEP 1: build a Mo-doped Ni(111) surface, adsorb ethane, relax to the 0 K
ground state, and compute the adsorption energy.

Writes outputs/relaxed.traj, which steps 2 and 3 read.

The OC20 task is not trained on isolated molecules, so asking it for free
ethane gives an unreliable number. Instead we relax a second slab+ethane
system with the molecule parked far above the surface. The model still sees
a slab, and the slab energy cancels in E_together - E_far.

Everything is run twice: once with plain UMA, once with Grimme D3 added to
the relaxation. OC20 is RPBE, which has no dispersion term, and alkane-metal
binding is almost entirely dispersion - so the uncorrected run is expected
to give roughly zero. The pair of numbers is the result, not either alone.
"""

import sys
from pathlib import Path

from ase.build import add_adsorbate, fcc111, molecule
from ase.calculators.mixing import SumCalculator
from ase.constraints import FixAtoms
from ase.optimize import BFGS

from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

SEPARATION = 8.0        # height used for the non-interacting reference


def new_calculator(with_d3=False):
    """A fresh calculator for each system.

    One calculator object cannot be reused across systems with different
    atom counts - it caches a results buffer sized to the first structure
    it sees and then raises a numpy broadcast error.

    with_d3 adds Grimme D3 as a second additive calculator, so dispersion
    contributes forces during relaxation and the molecule can settle at a
    realistic height. Evaluating D3 only as a single point on a UMA-relaxed
    geometry would miss most of the well, since D3 goes as R^-6 and the
    uncorrected molecule sits about 0.6 A too far out.

    RPBE parameters, because that is the functional OC20 was trained on.
    """
    unit = load_predict_unit(path=str(config.MODEL_PATH), device=config.DEVICE)
    uma = FAIRChemCalculator(unit, task_name=config.TASK_NAME)
    if not with_d3:
        return uma
    d3 = TorchDFTD3Calculator(damping="zero", xc="rpbe", device=config.DEVICE)
    return SumCalculator([uma, d3])


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


def run(with_d3, traj=None):
    """Relax the adsorbed system and its far reference.

    Returns (E, contact, cc): the adsorption energy, the closest
    adsorbate-metal distance after relaxation, and the C-C bond length.
    """
    label = "with D3" if with_d3 else "uncorrected"
    print(f"\n--- {label} ---")

    slab, ads = build(config.ADS_HEIGHT)
    slab.calc = new_calculator(with_d3)
    print(f"closest contact at start   = {gap(slab, ads):.2f} A  (want > 2.0)")
    E_together = relax(slab, f"surface + ethane ({label})", traj=traj)

    far, far_ads = build(SEPARATION)
    far.calc = new_calculator(with_d3)
    far_start = gap(far, far_ads)
    print(f"reference contact at start = {far_start:.2f} A  "
          f"(should be about {SEPARATION:.1f})")
    if far_start < SEPARATION - 1.0:
        raise RuntimeError(
            f"reference built at only {far_start:.2f} A - wrong height was "
            "passed to build(), so this is not a valid reference")

    E_far = relax(far, f"far reference ({label})")
    if gap(far, far_ads) < SEPARATION - 1.0:
        raise RuntimeError("the reference ethane drifted onto the slab")

    carbons = [i for i in ads if slab[i].symbol == "C"]
    cc = slab.get_distance(carbons[0], carbons[1], mic=True)

    print(f"surface + ethane           = {E_together:9.3f} eV")
    print(f"slab + distant ethane      = {E_far:9.3f} eV")

    return E_together - E_far, gap(slab, ads), cc


def main():
    config.check_checkpoint()

    # Plain UMA. This is the trajectory steps 2 and 3 read.
    E_ads, contact_ads, cc_ads = run(False)

    # Dispersion inside the relaxation loop, so the molecule can move in.
    E_d3, contact_d3, cc_d3 = run(True, traj=str(config.RELAXED_TRAJ))

    print("\n" + "=" * 52)
    print(f"{'':<22}{'uncorrected':>13}{'with D3':>13}")
    print(f"{'E_ads          [eV]':<22}{E_ads:>13.4f}{E_d3:>13.4f}")
    print(f"{'closest contact [A]':<22}{contact_ads:>13.2f}{contact_d3:>13.2f}")
    print(f"{'C-C bond        [A]':<22}{cc_ads:>13.2f}{cc_d3:>13.2f}")
    print("=" * 52)
    print("expect C-C ~1.53 A; dispersion should pull contact from ~4.1 to ~3.5 A")

    if cc_d3 > 2.0:
        print("\nWARNING: the C-C bond broke during relaxation")
    if E_d3 < -0.5:
        print("\nstrongly bound (chemisorbed)")
    elif E_d3 < 0:
        print("\nweakly bound (physisorbed)")
    else:
        print("\nnot bound - ethane prefers the gas phase")

    return E_d3


if __name__ == "__main__":
    main()