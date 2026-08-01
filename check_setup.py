"""
Run this FIRST. Verifies the environment before you waste time on a long job.

    python check_setup.py
"""

import sys

print(f"Python {sys.version_info.major}.{sys.version_info.minor}."
      f"{sys.version_info.micro}")
if sys.version_info < (3, 10):
    print("  FAIL - need Python 3.10+ (torch-geometric >= 2.7 dropped 3.9)")
    sys.exit(1)
print("  ok")

failed = False
for name in ["numpy", "polars", "torch", "ase", "fairchem.core"]:
    try:
        __import__(name)
        print(f"import {name:16s} ok")
    except ImportError as e:
        print(f"import {name:16s} FAIL - {e}")
        failed = True

if failed:
    print("\nSome imports failed. Try:  pip install -r requirements.txt")
    sys.exit(1)

import torch
print(f"\ntorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("  No GPU. Steps 0 and 4 are fine; steps 1-3 will be SLOW.")
    print("  Consider running the atomistic steps in Google Colab instead.")

import config
print(f"\ncheckpoint path : {config.MODEL_PATH}")
print(f"  exists        : {config.MODEL_PATH.exists()}")
print(f"telemetry       : {config.PARQUET}")
print(f"  exists        : {config.PARQUET.exists()}")

if config.MODEL_PATH.exists() and config.PARQUET.exists():
    print("\nEverything is in place. Run:  python run_all.py 0")
else:
    print("\nAdd the missing files to data/ before running the pipeline.")
