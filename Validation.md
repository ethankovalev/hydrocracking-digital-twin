# Validation Report: ethane adsorption on Mo-doped Ni(111)

*A record of how the adsorption energy in this pipeline was computed,
what was cross-checked, and how far the result can be trusted. Written to
be reusable: the same six-step protocol applies to any foundation-MLIP
result, with this system as the worked example.*

## System

- **Model / checkpoint:** `uma-s-1p2`
- **Task:** `oc20`
- **Trained reference method:** RPBE (no dispersion / van der Waals term)
- **System studied:** ethane on Mo-doped Ni(111), 4x4x4 slab, 3 Mo atoms
  substituted into the top layer
- **Property computed:** adsorption (binding) energy, `E_ads`

## 1. Is the reference protocol itself valid?

The OC20 task is not trained on isolated molecules, so a direct gas-phase
calculation of free ethane is unreliable. Instead, the reference is a
second relaxation of the same slab+ethane system with the molecule placed
far above the surface (`SEPARATION = 8.0 A`), so the slab contribution
cancels in `E_ads = E_together - E_far` and the model only ever sees a
slab+adsorbate system.

**Cross-checks performed:**

- [x] **Stable across independent starting geometries.** The reference
      was rerun at 8 A, 6 A, and 1 A initial separation. All three
      converged to the same `E_ads` (0.0032-0.0065 eV) and the same C-C
      bond length (1.53 A), confirming the far system is genuinely
      non-interacting and the result does not depend on starting height.
- [x] **Agrees with an independently-reasoned alternative method.** A
      direct molecule-in-a-box calculation (ethane in an empty
      ~15x15x15 A cell) gave the same `E_ads` and C-C length as the
      far-separation reference, confirming the two approaches are
      equivalent for this system.
- [ ] **Cell-size convergence not yet confirmed.** See section 5.

**Verdict:** the reference protocol is validated. Three independent
starting separations and one methodologically distinct alternative all
agree to within a few meV.

## 2. Is the result distinguishable from the model's own noise floor?

**Model's benchmarked MAE:** 0.84-0.90 kJ/mol (approx. 0.0087-0.0093 eV)
against reference DFT (PBE-D3), across a 38-compound benchmark
(Gharakhanyan et al., *FastCSP*, arXiv:2508.02641).

**Measured result (uncorrected):** 0.0032-0.0065 eV.

**Verdict: below the noise floor.** The uncorrected `E_ads` is smaller
than the model's own benchmarked agreement with the DFT it was trained to
reproduce. It should be read as *not resolvable from zero*, not as a
confirmed weak-binding measurement. The separation sweep in section 1
establishes that the protocol is sound; it does not establish that the
number is resolvable.

## 3. Are known model limitations relevant to this system?

| Known limitation | Relevant here? | Evidence |
|---|---|---|
| RPBE (the OC20 training functional) contains no dispersion term | **Yes** | Alkane-metal binding on a bare metal is almost entirely dispersion. Uncorrected result is indistinguishable from zero; the molecule relaxes *outward* to 4.10 A rather than settling into a well. |
| Accuracy degrades where underrepresented chemistry forms the structural core governing the interaction being measured (*ibid.*) | Not established | Metallic Ni-Mo surfaces with hydrocarbon adsorbates are within OC20's stated domain, but no system-specific check was run. |
| D3 corrections overbind on metal surfaces | **Yes** | Applies directly to the correction in section 4. See caveat below. |

## 4. Correction applied

- **Correction:** Grimme D3, zero damping, RPBE parameters, via
  `torch-dftd`.
- **Method:** applied **inside the relaxation loop** (as an additive
  calculator alongside UMA), not as a single-point correction on the
  uncorrected geometry.

This distinction is not cosmetic. D3 scales as R^-6, so evaluating it on
a geometry relaxed *without* dispersion misses most of the well. Measured
directly: the single-point correction on the uncorrected 4.10 A structure
gave -0.11 eV, against -0.308 eV when dispersion was allowed to
contribute forces during relaxation, roughly a third of the well depth.

| | uncorrected | with D3 |
|---|---|---|
| `E_ads` [eV] | +0.0032 | -0.308 |
| closest adsorbate-metal contact [A] | 4.10 | 2.78 |
| C-C bond [A] | 1.53 | 1.53 |

Dispersion pulled the molecule in by roughly 1.3 A and produced a binding
energy inside the experimental range for ethane on Ni(111) (-0.25 to
-0.35 eV, TPD). The C-C bond is unchanged, confirming nothing dissociated
during either relaxation.

**Caveat on the correction:** D3 is known to overbind on metal surfaces.
Landing inside the experimental range is suggestive, not confirmatory:
some fraction of -0.308 eV may be an artifact of the correction rather
than physical binding.

## 5. Remaining limitations

- **Contact distance not yet decomposed.** The 2.78 A figure is the
  closest *any* adsorbate atom to *any* metal atom. If that atom is a
  hydrogen pointing down, the carbon height is a normal 3.3-3.6 A; if the
  carbons themselves sit that close, it is evidence of D3 overbinding.
  Not yet resolved: measuring carbon height above the surface plane
  directly would settle it.
- **Single site and orientation.** Only ontop-Mo with one molecular
  orientation was relaxed. This is not a minimum over configuration
  space; a proper treatment would sample multiple sites and orientations
  and take the lowest.
- **Coverage not confirmed converged.** The 4x4 supercell gives roughly
  10 A between periodic images of the adsorbate (centre to centre), which
  should be adequate for a physisorbed species, but this has not been
  verified against a larger cell.
- **0 K electronic energy only.** No zero-point energy or entropy
  corrections.
- **Metallic catalyst model.** Industrial hydrocracking catalysts are
  sulfided (NiMoS2) on alumina with a zeolite cracking function. The
  metallic model sits inside OC20's training distribution; a sulfide
  would not.

## 6. Reproduction

```bash
python src/step1_build_relax.py
```

Prints both the uncorrected and D3-corrected `E_ads`, along with the
diagnostics used above: contact distance at start and after relaxation,
reference separation check, and C-C bond length. Requires the gated UMA
checkpoint (see README).

## Summary

Uncorrected, `uma-s-1p2` on the `oc20` task predicts essentially no
binding for ethane on Mo-doped Ni(111), a result below the model's own
benchmarked error against DFT and consistent with RPBE containing no
dispersion term. Adding RPBE-D3 inside the relaxation loop recovers
-0.308 eV and pulls the molecule 1.3 A closer to the surface, within the
experimental range. The reference protocol behind both numbers was
validated against three starting separations and one independent method.
The single most important caveat: D3 is known to overbind on metals, so
agreement with experiment is suggestive rather than confirmatory.