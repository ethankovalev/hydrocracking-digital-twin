"""
STEP 2: heat the relaxed structure to reactor temperature and check whether
the ethane stays on the surface.

Langevin dynamics adds two terms to Newton's second law:

    m dv/dt = F(r) - gamma*m*v + R(t)
              physics  friction   random kicks

Friction and noise together stand in for a heat bath we do not simulate.
Here the bath is the metal only. Friction applied to the ethane would damp
its centre-of-mass motion, which is exactly what desorption consists of -
we would be measuring the thermostat rather than the chemistry.

The cost of that choice is that the molecule now relies on physical contact
with the slab to reach 675 K, so its temperature is monitored separately.

The calculator must include D3, matching step 1. Plain UMA/OC20 gives no
binding well for a physisorbed alkane, so without dispersion the molecule
leaves regardless of what the thermostat does and the run tests nothing.
"""

import sys
from pathlib import Path

import numpy as np
from ase import units
from ase.io import Trajectory, read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import config
from step1_build_relax import new_calculator


def main():
    config.check_checkpoint()
    if not config.RELAXED_TRAJ.exists():
        raise FileNotFoundError("Run step 1 first - outputs/relaxed.traj is missing.")

    atoms = read(str(config.RELAXED_TRAJ))      # last frame = the relaxed one
    atoms.calc = new_calculator(with_d3=True)

    # Step 1 tagged them: 2 = adsorbate, 0/1 = metal.
    tags = atoms.get_tags()
    carbons, hydrogens, metal = [], [], []
    for i in range(len(atoms)):
        if tags[i] == 2 and atoms[i].symbol == "C":
            carbons.append(i)
        elif tags[i] == 2 and atoms[i].symbol == "H":
            hydrogens.append(i)
        elif tags[i] != 2:
            metal.append(i)

    if len(carbons) != 2:
        raise ValueError(f"expected 2 tagged carbons, found {len(carbons)}")

    # Decide once, at 0 K, which hydrogen belongs to which carbon.
    ch_pairs = []
    for h in hydrogens:
        d0 = atoms.get_distance(h, carbons[0])
        d1 = atoms.get_distance(h, carbons[1])
        ch_pairs.append((carbons[0], h) if d0 < d1 else (carbons[1], h))

    # Confirm we are reading the D3 geometry and not a stale uncorrected one.
    surface_z = max(atoms.positions[i, 2] for i in metal)
    lowest_c = min(atoms.positions[c, 2] for c in carbons)
    print(f"starting C-surface gap = {lowest_c - surface_z:.2f} A")
    print("(expect ~3.3 A from the D3 run; ~4.5 A means you are reading the")
    print(" uncorrected trajectory and there is no well to test)\n")

    # Count frozen atoms for the temperature-fluctuation estimate.
    n_frozen = 0
    for c in atoms.constraints:
        if hasattr(c, "index"):
            n_frozen += len(c.index)
    n_free = len(atoms) - n_frozen

    MaxwellBoltzmannDistribution(atoms, temperature_K=config.TEMPERATURE)

    temperatures, ads_temps, cc_bonds, ch_bonds, gaps = [], [], [], [], []

    def temperature_of(indices):
        """Instantaneous temperature of a subset of atoms."""
        p = atoms.get_momenta()
        m = atoms.get_masses()
        ekin = 0.0
        for i in indices:
            ekin += 0.5 * (p[i] ** 2).sum() / m[i]
        return 2 * ekin / (3 * len(indices) * units.kB)

    def sample():
        temperatures.append(atoms.get_temperature())
        ads_temps.append(temperature_of(carbons + hydrogens))
        cc_bonds.append(atoms.get_distance(carbons[0], carbons[1], mic=True))

        longest_ch = 0.0
        for c, h in ch_pairs:
            d = atoms.get_distance(c, h, mic=True)
            if d > longest_ch:
                longest_ch = d
        ch_bonds.append(longest_ch)

        # Plain z difference, NOT a minimum-image distance. The cell is
        # periodic in z, so mic would wrap a departing molecule back towards
        # the underside of the slab and report it as still adsorbed.
        z_surf = max(atoms.positions[i, 2] for i in metal)
        z_c = min(atoms.positions[c, 2] for c in carbons)
        gaps.append(z_c - z_surf)

    # Thermostat the metal only.
    friction = np.zeros((len(atoms), 1))
    for i in metal:
        friction[i, 0] = config.FRICTION_PER_FS / units.fs

    md = Langevin(
        atoms,
        timestep=config.TIMESTEP_FS * units.fs,
        temperature_K=config.TEMPERATURE,
        friction=friction,
        fixcm=False,            # the bottom layers are already frozen
    )

    print(f"warming up ({config.WARMUP_STEPS} steps, discarded)...")
    md.run(config.WARMUP_STEPS)

    print(f"production ({config.RUN_STEPS} steps)...")
    traj = Trajectory(str(config.MD_TRAJ), "w", atoms)
    for _ in range(config.RUN_STEPS // config.SAMPLE_EVERY):
        md.run(config.SAMPLE_EVERY)
        sample()
        traj.write()
        if gaps[-1] > 8.0:
            print("ethane is over 8 A off the surface - stopping early")
            print("(further out it would cross the periodic z boundary and")
            print(" re-adsorb on the frozen underside of the slab)")
            break
    traj.close()

    T = np.array(temperatures)
    T_ads = np.array(ads_temps)
    cc = np.array(cc_bonds)
    ch = np.array(ch_bonds)
    gap = np.array(gaps)

    # Instantaneous T fluctuates by ~sqrt(2/3N) in the canonical ensemble.
    # Judge the thermostat by the MEAN, not by individual frames.
    wobble = config.TEMPERATURE * np.sqrt(2 / (3 * n_free))

    print(f"\ntemperature (all)    : {T.mean():.0f} K  (target {config.TEMPERATURE:.0f} K)")
    print(f"    wobble           : +/- {T.std():.0f} K  (+/- {wobble:.0f} K is normal)")
    print(f"temperature (ethane) : {T_ads.mean():.0f} K")
    print(f"C-C bond             : {cc.mean():.2f} A  (max {cc.max():.2f}, expect ~1.53)")
    print(f"C-H bond             : {ch.mean():.2f} A  (max {ch.max():.2f}, expect ~1.09)")
    print(f"C-surface gap        : start {gap[0]:.2f} A, end {gap[-1]:.2f} A, "
          f"max {gap.max():.2f} A")

    print("")

    # These are independent failures - check all of them, not if/elif.
    if abs(T_ads.mean() - config.TEMPERATURE) > 3 * wobble:
        print("-> the ethane is NOT in equilibrium with the slab. Energy is not")
        print("   crossing the interface fast enough. Put a small friction on")
        print("   the adsorbate as well and rerun.")

    if ch.max() > 1.6:
        print("-> a C-H bond broke. Check TIMESTEP_FS is 0.5 or less before")
        print("   blaming the chemistry - too large a step breaks C-H first.")

    if cc.max() > 2.0:
        print("-> C-C bond broke. Suspicious on this timescale: a 1 eV barrier")
        print("   at 675 K takes ~3 us to cross by chance, a million times longer")
        print("   than this run. More likely the potential than real chemistry.")

    # Judge desorption on the last quarter, not on one stray frame.
    tail = gap[3 * len(gap) // 4:]
    if tail.mean() > 6.0:
        print("-> ethane desorbed. This is now the SURPRISING result: with the")
        print("   D3-corrected E_ads of -0.31 eV and kB*T = 0.058 eV at 675 K,")
        print("   the residence time should be tens of ps, longer than this run.")
    elif cc.max() <= 2.0 and ch.max() <= 1.6:
        print("-> ethane stayed put, which is what -0.31 eV predicts at 675 K.")
        print("   The 0 K structure survives at reactor temperature.")


if __name__ == "__main__":
    main()
