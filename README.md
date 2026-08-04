# Physics-Informed Digital Twin for Refinery Hydrocracking

Links atomistic catalyst simulation to plant-scale yield prediction. A
machine-learned interatomic potential computes the activation energy for
C–C scission on a Ni–Mo surface; that barrier then constrains a neural
network trained on reactor telemetry.

```
SCADA telemetry ──┐
                  ├──> physics-informed NN ──> conversion & yield
Ni-Mo/C2H6 Ea ────┘
```

## The idea

A neural network fitted to plant data has no reason to respect chemistry.
This pipeline computes an activation energy from first principles and uses
it as a **constraint** on the network's temperature dependence, instead of
letting the network infer kinetics from data alone.

The two layers run on completely different cadences:

| Layer | Runs | Produces |
|---|---|---|
| Atomistic (steps 1–3) | a handful of times, offline | Ea — one number |
| Data-driven (step 4) | every telemetry row | conversion, yield |

## Requirements

**Python 3.10 or newer.** Not optional: `fairchem-core` depends on
`torch-geometric >= 2.7`, which dropped Python 3.9. On 3.9 pip reports a
confusing "no matching distribution found" error rather than saying this.

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip          # do NOT skip - old pip resolves badly
pip install -r requirements.txt
python check_setup.py              # verify before running anything
```

### Model weights

The UMA checkpoint is gated. Request access at
<https://huggingface.co/facebook/UMA>, then `huggingface-cli login`.
Put `uma-s-1p2.pt` in `data/`, or set `UMA_CHECKPOINT` to its path.

### Data

Put your telemetry at `data/reactor_data.parquet` (or
`data/sample_data.csv` — step 0 converts it).

## Running

```bash
python check_setup.py      # always start here
python run_all.py 0        # clean the telemetry
python run_all.py          # the whole pipeline
python run_all.py 0 4      # skip the slow atomistic steps
```

| Step | What it does |
|---|---|
| 0 | Filter bad sensor reads, convert to SI, sort by time |
| 1 | Mo-doped Ni(111) slab + ethane, BFGS relaxation, E_ads |
| 2 | Langevin MD at 675 K — does the 0 K structure survive? |
| 3 | Climbing-image NEB → activation energy |
| 4 | Arrhenius-constrained network vs. a plain baseline |

## Hardware

Steps 1–3 call a GNN thousands of times and want a CUDA GPU. They run on a
laptop CPU but slowly — a 10 ps MD run may take hours. Steps 0 and 4 are
CPU-friendly. If you have no GPU, run the atomistic steps in Google Colab.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `no matching distribution for torch-geometric>=2.7.0` | Python 3.9. Rebuild the venv with 3.10+ |
| `conflicting dependencies` between torch and ase | Hand-pinned torch in requirements.txt fighting fairchem. Don't pin it |
| pip downloads a dozen versions of one package | Old pip. `pip install --upgrade pip` |
| `could not broadcast input array from shape (N,) into shape (M,)` | One calculator reused across systems with different atom counts. Build a fresh one per system |
| Physics loss reads exactly `0.00000` from epoch 0 | A `torch.clamp` upstream of the gradient. Clamp has zero gradient outside its bounds |

## Running with Docker

    docker build -t hydrocracking .
    docker run --rm \
      -v "$(pwd)/data:/app/data" \
      -v "$(pwd)/outputs:/app/outputs" \
      hydrocracking python run_all.py 0

Place `reactor_data.parquet` and the UMA checkpoint in `data/` first.
The checkpoint requires access from https://huggingface.co/facebook/UMA

## Known simplifications

- The catalyst is modelled as a **metallic** Ni–Mo surface. Industrial
  hydrocracking catalysts are sulfided (NiMoS₂) on alumina with a zeolite
  cracking function. The metallic model at least sits inside the training
  distribution of an OC20-trained potential; a sulfide would not.
- Step 4 assumes a **first-order** reaction. The Arrhenius constraint uses
  only the temperature derivative, so the unknown pre-exponential and space
  time cancel — but the reaction order itself is an assumption.
- MD in step 2 runs for picoseconds; a 1 eV barrier takes microseconds to
  cross by chance. Bond breaking during MD indicates model failure, not
  chemistry. Barriers come from step 3.
- E_ads should be sanity-checked against a known system before the absolute numbers are trusted. The current ethane result is E_ads = 0.0032 eV (unbound) — plausibly because RPBE has no dispersion term and physisorption here is dispersion-dominated, but this hasn't been confirmed externally yet. A chemisorbed reference (e.g. CO on Pt(111), literature ≈ −1.37 ± 0.13 eV) tests the model's chemisorption behaviour but isn't the right comparison for a physisorbed system like this...

## References

- Chanussot et al., *Open Catalyst 2020 (OC20) Dataset and Community
  Challenges*, ACS Catalysis 11, 6059 (2021)
- Wander, Shuaibi, Kitchin, Ulissi, Zitnick, *CatTSunami: Accelerating
  Transition State Energy Calculations with Pre-trained Graph Neural
  Networks*, arXiv:2405.02078 (2024)
- Meta FAIR Chemistry, *UMA: Universal Models for Atoms*
