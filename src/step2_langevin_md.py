"""
STEP 2: heat the relaxed structure to reactor temperature and check whether
the ethane stays on the surface.

Langevin dynamics adds two terms to Newton's second law:

    m dv/dt = F(r) - gamma*m*v + R(t)
              physics  friction   random kicks

Friction and noise together stand in for a heat bath we do not simulate -
the same idea as lumping convective cooling into a heat transfer
coefficient. Their relative strength is fixed by the fluctuation-dissipation
theorem, which is why you only specify T and gamma.
"""

import sys
from pathlib import Path

import numpy as np
from ase import units
from ase.io import Trajectory, read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def main():
    config.check_checkpoint()
    if not config.RELAXED_TRAJ.exists():
        raise FileNotFoundError("Run step 1 first - outputs/relaxed.traj is missing.")

    atoms = read(str(config.RELAXED_TRAJ))      # last frame = the relaxed one
    unit = load_predict_unit(path=str(config.MODEL_PATH), device=config.DEVICE)
    atoms.calc = FAIRChemCalculator(unit, task_name=config.TASK_NAME)

    # Step 1 tagged them: 2 = adsorbate, 0/1 = metal.
    tags = atoms.get_tags()
    carbons = [i for i in range(len(atoms))
               if tags[i] == 2 and atoms[i].symbol == "C"]
    metal = [i for i in range(len(atoms)) if tags[i] != 2]

    # Initial velocities from the Maxwell-Boltzmann distribution. Constraints
    # are applied automatically, so frozen atoms get zero velocity.
    MaxwellBoltzmannDistribution(atoms, temperature_K=config.TEMPERATURE)

    temperatures, cc_bonds, gaps = [], [], []

    def sample():
        temperatures.append(atoms.get_temperature())
        cc_bonds.append(atoms.get_distance(carbons[0], carbons[1], mic=True))
        gaps.append(min(atoms.get_distance(c, m, mic=True)
                        for c in carbons for m in metal))

    md = Langevin(
        atoms,
        timestep=config.TIMESTEP_FS * units.fs,
        temperature_K=config.TEMPERATURE,
        friction=config.FRICTION_PER_FS / units.fs,
    )

    print(f"warming up ({config.WARMUP_STEPS} steps, discarded)...")
    md.run(config.WARMUP_STEPS)

    print(f"production ({config.RUN_STEPS} steps)...")
    md.attach(sample, interval=config.SAMPLE_EVERY)
    md.attach(Trajectory(str(config.MD_TRAJ), "w", atoms).write,
              interval=config.SAMPLE_EVERY)
    md.run(config.RUN_STEPS)

    T = np.array(temperatures)
    cc = np.array(cc_bonds)
    gap = np.array(gaps)

    n_frozen = sum(len(c.get_indices()) for c in atoms.constraints)
    n_free = len(atoms) - n_frozen

    # Instantaneous T fluctuates by ~sqrt(2/3N) in the canonical ensemble.
    # Judge the thermostat by the MEAN, not by individual frames.
    wobble = config.TEMPERATURE * np.sqrt(2 / (3 * n_free))

    print(f"\ntemperature : {T.mean():.0f} K  (target {config.TEMPERATURE:.0f} K)")
    print(f"    wobble  : +/- {T.std():.0f} K  (+/- {wobble:.0f} K is normal)")
    print(f"C-C bond    : {cc.mean():.2f} A  (should stay near 1.5 A)")
    print(f"C-metal gap : {gap.mean():.2f} A  (largest {gap.max():.2f} A)")

    if cc.max() > 2.0:
        print("\n-> C-C bond broke. Suspicious on a 5 ps timescale: a 1 eV")
        print("   barrier takes milliseconds to cross by chance, so this is")
        print("   more likely the ML potential failing than real chemistry.")
    elif gap.max() > 5.0:
        print("\n-> ethane desorbed. Expected if E_ads is small: kB*T at 675 K")
        print("   is ~0.058 eV, comparable to a weak physisorption energy.")
    else:
        print("\n-> ethane stayed put. The 0 K structure survives at 675 K.")


if __name__ == "__main__":
    main()
