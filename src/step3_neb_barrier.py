"""
STEP 3: find the activation energy for breaking the C-C bond.

WHAT HAPPENS HERE, IN FOUR STEPS:

  1. Load the START picture   - ethane sitting on the surface (from step 1)
  2. Build the END picture    - two CH3 pieces sitting on the surface
  3. Ask ASE to find the path between them
  4. The highest energy along that path is the activation energy

Everything else in this file is either printing diagnostics or checking
that something has not gone wrong.
"""

import sys
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import BFGS, FIRE

try:
    from ase.mep import NEB, NEBTools
except ImportError:
    from ase.neb import NEB, NEBTools

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

FMAX_NEB = getattr(config, "FMAX_NEB", 0.10)


# ===========================================================================
# Small helpers - each does one thing
# ===========================================================================

def find_parts(atoms):
    """Who is what. Returns three lists of atom numbers.

    Step 1 labelled every atom: tag 2 = the molecule, anything else = metal.
    """
    tags = atoms.get_tags()
    carbons = [i for i in range(len(atoms))
               if tags[i] == 2 and atoms[i].symbol == "C"]
    hydrogens = [i for i in range(len(atoms))
                 if tags[i] == 2 and atoms[i].symbol == "H"]
    metal = [i for i in range(len(atoms)) if tags[i] != 2]
    return carbons, hydrogens, metal


def split_into_methyls(atoms):
    """Work out which atoms belong to each CH3 half of the ethane.

    Each hydrogen joins whichever carbon it is closest to.
    Returns two lists: [C, H, H, H] and [C, H, H, H].
    """
    carbons, hydrogens, _ = find_parts(atoms)
    left = [carbons[0]]
    right = [carbons[1]]

    for h in hydrogens:
        if (atoms.get_distance(h, carbons[0], mic=True)
                < atoms.get_distance(h, carbons[1], mic=True)):
            left.append(h)
        else:
            right.append(h)
    return left, right


def describe(atoms, label):
    """Print what this structure looks like, in plain English."""
    carbons, hydrogens, metal = find_parts(atoms)

    cc = atoms.get_distance(carbons[0], carbons[1], mic=True)
    c_to_metal = min(atoms.get_distance(c, m, mic=True)
                     for c in carbons for m in metal)
    loosest_h = max(min(atoms.get_distance(h, c, mic=True) for c in carbons)
                    for h in hydrogens)

    print(f"\n  {label}")
    print(f"    C-C distance          {cc:5.2f} A   "
          f"({'intact ethane' if cc < 2.0 else 'broken apart'})")
    print(f"    closest C to metal    {c_to_metal:5.2f} A   "
          f"({'bonded' if c_to_metal < 2.6 else 'floating above'})")
    print(f"    furthest H from its C {loosest_h:5.2f} A   "
          f"({'attached' if loosest_h < 1.4 else 'H HAS COME OFF'})")
    return cc, c_to_metal


def relax(atoms, fmax, label, maxstep=0.1):
    """Settle a structure into its nearest low-energy shape.

    Uses FIRE first if the starting forces are large. FIRE copes with bad
    starting geometries; BFGS is more precise but gets confused by them.
    """
    force = np.linalg.norm(atoms.get_forces(apply_constraint=False), axis=1).max()
    print(f"\n  relaxing {label}: starting force = {force:.1f} eV/A")

    if force > 10.0:
        print("    that is large - loosening it up with FIRE first")
        FIRE(atoms, logfile="-").run(fmax=1.0, steps=200)

    opt = BFGS(atoms, logfile="-", maxstep=maxstep)
    opt.run(fmax=fmax, steps=300)

    if not opt.converged():
        print(f"    WARNING: {label} did not settle down")
    return opt.converged()


# ===========================================================================
# The four steps
# ===========================================================================

def main():
    config.check_checkpoint()
    if not config.RELAXED_TRAJ.exists():
        raise FileNotFoundError("Run step 1 first - outputs/relaxed.traj is missing.")

    unit = load_predict_unit(path=str(config.MODEL_PATH), device=config.DEVICE)

    def calculator():
        return FAIRChemCalculator(unit, task_name=config.TASK_NAME)

    # -- STEP 1: the starting picture -----------------------------------
    print("=" * 60)
    print("  STEP 1 of 4: load the starting structure")
    print("=" * 60)

    start = read(str(config.RELAXED_TRAJ))
    start.calc = calculator()
    describe(start, "START (ethane on the surface)")

    # -- STEP 2: the ending picture -------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 2 of 4: build the broken-apart structure")
    print("=" * 60)

    end = start.copy()          # copy, so the atoms stay in the same order
    end.calc = calculator()

    left, right = split_into_methyls(start)
    c_left, c_right = left[0], right[0]

    # Which way to pull them apart: along the line joining the two carbons,
    # flattened so they slide sideways instead of lifting off the surface.
    direction = start.positions[c_right] - start.positions[c_left]
    direction[2] = 0.0
    direction = direction / np.linalg.norm(direction)   # length exactly 1

    # Move each half in opposite directions, half the needed distance each.
    current_gap = start.get_distance(c_left, c_right, mic=True)
    move_by = (config.SEPARATION - current_gap) / 2.0
    end.positions[left] -= direction * move_by
    end.positions[right] += direction * move_by

    # Now set both halves down onto the metal so they can actually bond.
    _, _, metal = find_parts(start)
    surface_height = max(start.positions[i, 2] for i in metal)
    target_height = surface_height + config.CH3_BOND_HEIGHT

    for half, carbon in [(left, c_left), (right, c_right)]:
        drop = target_height - end.positions[carbon, 2]
        end.positions[half, 2] += drop

    relax(end, config.FMAX, "the broken-apart structure")
    end_cc, _ = describe(end, "END (two CH3 pieces on the surface)")

    if end_cc < 2.0:
        raise RuntimeError(
            f"The two halves snapped back together (C-C = {end_cc:.2f} A). "
            "They are not sticking to the metal. If E_ads from step 1 was "
            "near zero, this reaction cannot happen on this surface."
        )

    # -- STEP 3: the path between them ----------------------------------
    print("\n" + "=" * 60)
    print("  STEP 3 of 4: find the path from START to END")
    print("=" * 60)

    # A chain of in-between pictures. ASE moves them all at once until they
    # trace the easiest route from START to END.
    path = [start]
    for _ in range(config.N_IMAGES):
        picture = start.copy()
        picture.calc = calculator()
        path.append(picture)
    path.append(end)

    neb = NEB(path, climb=True, method="improvedtangent")
    neb.interpolate(method="idpp")      # sensible first guess for the middle

    print(f"\n  moving {len(path)} pictures at once (target {FMAX_NEB} eV/A)")
    opt = BFGS(neb, trajectory=str(config.NEB_TRAJ), logfile="-", maxstep=0.05)
    opt.run(fmax=FMAX_NEB, steps=300)
    settled = opt.converged()

    # -- STEP 4: read off the barrier -----------------------------------
    print("\n" + "=" * 60)
    print("  STEP 4 of 4: the activation energy")
    print("=" * 60)

    Ea, reaction_energy = NEBTools(path).get_barrier()
    energies = [p.get_potential_energy() for p in path]
    uphill = [e - energies[0] for e in energies]
    highest = int(np.argmax(uphill))

    print("\n  energy of each picture along the path (eV above the start):")
    for i, e in enumerate(uphill):
        bar = "#" * max(0, int(e * 20))
        note = "  <- highest point" if i == highest else ""
        print(f"    picture {i}: {e:6.2f}  {bar}{note}")

    print(f"\n  activation energy  = {Ea:.3f} eV")
    print(f"  reaction energy    = {reaction_energy:.3f} eV")
    print(f"  path settled down  = {settled}")

    # -- is this number believable? --------------------------------------
    print("\n  sanity checks:")
    if highest in (0, len(path) - 1):
        print("    FAIL: the highest point is one of the two ends, so there is")
        print("          no hill in between. Nothing to measure.")
    else:
        print("    OK: the highest point is in the middle, as it should be.")

    if Ea < 0.1:
        print("    FAIL: breaking a C-C bond on a metal should cost ~1-2 eV.")
        print("          This is far too small to be real.")
    elif Ea > 5.0:
        print("    ODD: unusually large. Check the end structure makes sense.")
    else:
        print("    OK: the size of this barrier is physically plausible.")

    if not settled:
        print("    NOTE: the path never fully settled, so treat Ea as rough.")

    write(str(config.OUTPUT_DIR / "neb_path.xyz"), path)
    np.save(str(config.EA_FILE), np.array([Ea]))
    return Ea


if __name__ == "__main__":
    main()


