"""
All paths and constants in one place, so the scripts contain physics
rather than filesystem plumbing.
"""

import os
import sys
from pathlib import Path

# --- fail early and clearly on an unsupported Python ----------------------
# torch-geometric >= 2.7 (which fairchem-core requires) needs Python 3.10+.
# On 3.9 pip silently offers only torch-geometric <= 2.6.1 and then reports
# a confusing "no matching distribution" error.
if sys.version_info < (3, 10):
    raise RuntimeError(
        f"Python {sys.version_info.major}.{sys.version_info.minor} is too old.\n"
        "fairchem-core needs torch-geometric >= 2.7, which requires Python 3.10+.\n"
        "Install Python 3.13 from python.org, then rebuild the environment:\n"
        "    rm -rf .venv && python3.13 -m venv .venv\n"
        "    source .venv/bin/activate && pip install --upgrade pip\n"
        "    pip install -r requirements.txt"
    )

import torch  # noqa: E402  (imported after the version check on purpose)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_CSV = DATA_DIR / "sample_data.csv"
PARQUET = DATA_DIR / "reactor_data.parquet"
CLEAN_PARQUET = DATA_DIR / "reactor_data_clean.parquet"

RELAXED_TRAJ = OUTPUT_DIR / "relaxed.traj"
MD_TRAJ = OUTPUT_DIR / "md.traj"
NEB_TRAJ = OUTPUT_DIR / "neb.traj"
EA_FILE = OUTPUT_DIR / "Ea_eV.npy"
PINN_WEIGHTS = OUTPUT_DIR / "pinn.pt"

MODEL_PATH = Path(os.environ.get("UMA_CHECKPOINT", DATA_DIR / "uma-s-1p2.pt"))
DEVICE="cuda" if torch.cuda.is_available() else "cpu"



# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
R_GAS = 8.3145               # J/mol/K
EV_TO_J_PER_MOL = 96485.0   # 1 eV per molecule x Avogadro


# ---------------------------------------------------------------------------
# Atomistic model
# ---------------------------------------------------------------------------
SLAB_SIZE = (4, 4, 4)       # (nx, ny, layers)
VACUUM = 12.0               # Ang each side of the slab
N_FIXED_LAYERS = 2
ADSORBATE = "C2H6"
ADS_HEIGHT = 2.5            # Ang above the surface plane
FMAX = 0.05 
FMAX_NEB = 0.10               # eV/Ang relaxation convergence
TASK_NAME = "oc20"


# ---------------------------------------------------------------------------
# Molecular dynamics
# ---------------------------------------------------------------------------
TEMPERATURE = 648.0         # K - representative of the SCADA range
TIMESTEP_FS = 0.25           # set by the C-H stretch period (~11 fs)
FRICTION_PER_FS = 0.01      # 1/gamma = 100 fs thermal coupling
WARMUP_STEPS = 1000
RUN_STEPS = 5_000
SAMPLE_EVERY = 20
CH3_BOND_HEIGHT = 1.35 # a real Ni–C covalent bond, typically 1.8-1.9 Å. 


# ---------------------------------------------------------------------------
# NEB
# ---------------------------------------------------------------------------
N_IMAGES = 5                # intermediate images (plus 2 endpoints)
SEPARATION = 3.2            # Ang, final C-C separation


# ---------------------------------------------------------------------------
# PINN
# ---------------------------------------------------------------------------
FEATURES = [
    "wax_feed_tph", "feed_density_proxy", "fresh_h2_flow_Nm3h",
    "recycle_gas_flow_Nm3h", "quench1_flow_tph", "quench2_flow_tph",
    "temp_outlet_K", "pressure_Pa", "catalyst_age_days",
]
TEMP_COL = "temp_inlet_K"   # kept separate: we differentiate the net w.r.t. it
TARGETS = ["conversion_pct", "naphtha_yield_pct"]

PHYSICS_WEIGHT = 0.1
EPOCHS = 400
HIDDEN = 64
TRAIN_FRACTION = 0.8        # split by TIME, not randomly


def check_checkpoint():
    """Fail with a useful message rather than deep inside fairchem."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"UMA checkpoint not found at {MODEL_PATH}\n"
            "Request access at https://huggingface.co/facebook/UMA, run\n"
            "`huggingface-cli login`, then put uma-s-1p2.pt in data/ or set\n"
            "the UMA_CHECKPOINT environment variable to its path."
        )
