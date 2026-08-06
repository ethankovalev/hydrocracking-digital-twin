# Validation: ethane adsorption on Mo-doped Ni(111)

This document records how the adsorption energy in this pipeline was
computed, cross-checked, and what its limits are. It is intended to let
someone reproduce the same protocol on a different adsorbate or slab and
know how far to trust the result.

## Model

- Checkpoint: `uma-s-1p2`
- Task: `oc20`
- Trained functional: RPBE (no dispersion / van der Waals term)

## Reference protocol

The OC20 task is not trained on isolated molecules, so a direct gas-phase
calculation of free ethane is unreliable. Instead, the reference is a
second relaxation of the same slab+ethane system with the molecule placed
far above the surface (`SEPARATION = 8.0 Å`), so the slab contribution
cancels in `E_ads = E_together - E_far` and the model only ever sees a
slab+adsorbate system.

This was cross-checked two ways before being trusted:

- **Separation sweep**: the reference was rerun at 8 Å, 6 Å, and 1 Å. All
  three converged to the same E_ads (0.0032-0.0065 eV) and the same C-C
  bond length (1.53 Å), confirming the far system is genuinely
  non-interacting and the result does not depend on the starting height.
- **Molecule-in-a-box check**: a direct isolated-molecule calculation
  (ethane in an empty ~15x15x15 Å cell) gave the same E_ads and C-C length
  as the far-separation reference, confirming the two approaches are
  equivalent for this system.

## Resolution floor

UMA's own benchmarked accuracy against reference DFT (PBE-D3) is a mean
absolute error of 0.84-0.90 kJ/mol (~0.0087-0.0093 eV), reported across a
38-compound crystal structure benchmark (Gharakhanyan et al., FastCSP,
arXiv:2508.02641).

The uncorrected E_ads for ethane on Mo-doped Ni(111) is 0.003-0.007 eV -
**below this error floor**. This result should be read as "not resolvable
from zero," not as a confirmed weak-binding measurement. It reflects the
absence of a dispersion term in RPBE, not a resolved physical value.

## D3 correction

RPBE-D3 (Grimme, zero damping) was added via `torch-dftd`, applied
**inside the relaxation loop** (not as a single-point correction on the
uncorrected geometry). This matters: D3 scales as R^-6, and a single-point
correction on the uncorrected 4.10 Å geometry captures only ~40% of the
well depth compared to letting the geometry relax under the added forces.

| | uncorrected | with D3 |
|---|---|---|
| E_ads [eV] | +0.0032 | -0.308 |
| closest contact [Å] | 4.10 | 2.78 |
| C-C bond [Å] | 1.53 | 1.53 |

Adding D3 pulled the molecule in by ~1.3 Å and produced a binding energy
inside the experimental range for ethane on Ni(111) (-0.25 to -0.35 eV,
TPD).

## Caveats

- **D3 is known to overbind on metal surfaces.** Agreement with the
  experimental range above is suggestive, not confirmatory - some of the
  -0.308 eV may be an artifact of the correction rather than the true
  binding energy.
- **Single site and orientation.** Only one adsorption site (ontop-Mo) and
  one molecular orientation were relaxed. This is not a minimum over the
  configuration space; a proper treatment would sample multiple sites and
  orientations and take the lowest energy.
- **Coverage not converged.** The 3x3 unit cell places ethane ~3.5 Å from
  its own periodic image, which is not confirmed to be large enough for a
  dispersion-bound system.
- **0 K electronic energy only.** No zero-point energy or entropy
  corrections are included.
- **Metallic catalyst model.** Industrial hydrocracking catalysts are
  sulfided (NiMoS2) on alumina with a zeolite cracking function. The
  metallic model sits inside the OC20 training distribution; a sulfide
  would not.

## Reproduction

```bash
python src/step1_build_relax.py
```

Prints both the uncorrected and D3-corrected E_ads, along with the
diagnostics (contact distance, C-C bond length) used above to validate
the result.
