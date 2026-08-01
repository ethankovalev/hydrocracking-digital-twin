"""
STEP 4: the physics-informed neural network.

Where the atomistic layer and the plant telemetry meet.

  Ea    <- from NEB. ONE number, computed once, offline.
  T, P  <- from the parquet file. Thousands of rows, every training step.

The network learns conversion and yield from SCADA data (the data loss).
A second term forces its implied reaction rate to obey Arrhenius, with the
activation energy pinned to the DFT value (the physics loss).
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class YieldNet(nn.Module):
    """Outputs raw scores; conversion/yield are sigmoid(score)."""

    def __init__(self, n_features, t_mean, t_std, hidden=config.HIDDEN):
        super().__init__()
        self.t_mean, self.t_std = t_mean, t_std
        # Tanh rather than ReLU: the physics loss differentiates the network
        # and then backpropagates through that derivative. ReLU's derivative
        # is piecewise constant, which makes the physics gradient jumpy.
        self.net = nn.Sequential(
            nn.Linear(n_features + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def raw(self, x_norm, t_raw):
        # Temperature is normalised INSIDE forward, from raw Kelvin, so
        # autograd gives a derivative in physical units. Normalising outside
        # would silently scale the derivative by t_std.
        t_norm = (t_raw - self.t_mean) / self.t_std
        return self.net(torch.cat([x_norm, t_norm], dim=1))

    def forward(self, x_norm, t_raw):
        return torch.sigmoid(self.raw(x_norm, t_raw))


def physics_residual(model, x_norm, t_raw, Ea_J):
    """How badly does the network violate Arrhenius?

    First order:   X = 1 - exp(-k*tau)  =>  k*tau = -ln(1 - X)
    Arrhenius:     ln k = ln A - Ea/(R T)
    Differentiate: d(ln k)/dT = Ea/(R T^2)

    tau and A are unknown but CONSTANT, so they vanish under
    differentiation - we constrain the temperature sensitivity of the rate
    without ever needing the pre-exponential factor.

    NOTE on numerics: with X = sigmoid(z), the identity

        -ln(1 - X) = softplus(z)

    lets us compute ln(k) directly from the raw score z. The obvious
    alternative - clamping X away from 0 and 1 before taking logs - is a
    trap: torch.clamp has EXACTLY ZERO gradient outside its bounds, so a
    confident model produces a hard-zero physics gradient and this whole
    term silently stops doing anything.
    """
    t = t_raw.clone().requires_grad_(True)
    z = model.raw(x_norm, t)[:, 0:1]          # raw score for conversion

    ln_k = torch.log(F.softplus(z) + 1e-12)   # + eps keeps gradient alive

    dlnk_dT = torch.autograd.grad(
        ln_k, t,
        grad_outputs=torch.ones_like(ln_k),
        create_graph=True,                    # needed to backprop the derivative
    )[0]

    return dlnk_dT - Ea_J / (config.R_GAS * t ** 2)


def main():
    if not config.EA_FILE.exists():
        raise FileNotFoundError("Run step 3 first - outputs/Ea_eV.npy is missing.")
    if not config.CLEAN_PARQUET.exists():
        raise FileNotFoundError("Run step 0 first - the clean parquet is missing.")

    Ea_eV = float(np.load(str(config.EA_FILE))[0])
    Ea_J = Ea_eV * config.EV_TO_J_PER_MOL
    print(f"Ea from NEB = {Ea_eV:.3f} eV = {Ea_J/1000:.1f} kJ/mol")

    df = pl.read_parquet(config.CLEAN_PARQUET)
    print(f"loaded {len(df)} rows")

    X = df.select(config.FEATURES).to_numpy().astype(np.float32)
    T = df.select(config.TEMP_COL).to_numpy().astype(np.float32)
    Y = df.select(config.TARGETS).to_numpy().astype(np.float32) / 100.0

    # Split by TIME. A random split puts minute 499 in test and 500 in
    # training - the model memorises neighbours and looks better than it is.
    cut = int(config.TRAIN_FRACTION * len(X))
    Xtr, Xte = X[:cut], X[cut:]
    Ttr, Tte = T[:cut], T[cut:]
    Ytr, Yte = Y[:cut], Y[cut:]

    # Normalise with TRAINING statistics only.
    x_mean, x_std = Xtr.mean(0), Xtr.std(0) + 1e-8
    t_mean, t_std = float(Ttr.mean()), float(Ttr.std()) + 1e-8

    Xtr_t = torch.tensor((Xtr - x_mean) / x_std)
    Ttr_t = torch.tensor(Ttr)
    Ytr_t = torch.tensor(Ytr)
    Xte_t = torch.tensor((Xte - x_mean) / x_std)
    Tte_t = torch.tensor(Tte)
    Yte_t = torch.tensor(Yte)

    torch.manual_seed(0)
    model = YieldNet(len(config.FEATURES), t_mean, t_std)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\ntraining on {len(Xtr)} rows, testing on {len(Xte)}")
    for epoch in range(config.EPOCHS):
        opt.zero_grad()          # PyTorch accumulates gradients - clear first
        data_loss = ((model(Xtr_t, Ttr_t) - Ytr_t) ** 2).mean()
        phys_loss = (physics_residual(model, Xtr_t, Ttr_t, Ea_J) ** 2).mean()
        (data_loss + config.PHYSICS_WEIGHT * phys_loss).backward()
        opt.step()

        if epoch % 50 == 0 or epoch == config.EPOCHS - 1:
            print(f"  epoch {epoch:4d}  data {data_loss.item():.5f}  "
                  f"physics {phys_loss.item():.5f}")

    # --- baseline with no physics -----------------------------------------
    # If the physics term does not help, you have a neural network in a lab
    # coat. This comparison is the point of the whole pipeline.
    torch.manual_seed(0)         # same initialisation, only the loss differs
    base = YieldNet(len(config.FEATURES), t_mean, t_std)
    base_opt = torch.optim.Adam(base.parameters(), lr=1e-3)
    for _ in range(config.EPOCHS):
        base_opt.zero_grad()
        ((base(Xtr_t, Ttr_t) - Ytr_t) ** 2).mean().backward()
        base_opt.step()

    with torch.no_grad():
        pinn_mae = (model(Xte_t, Tte_t) - Yte_t).abs().mean(0) * 100
        base_mae = (base(Xte_t, Tte_t) - Yte_t).abs().mean(0) * 100

    print("\n" + "=" * 58)
    print("  held-out test MAE (percentage points)")
    print("                       conversion   naphtha yield")
    print(f"  physics-informed  : {pinn_mae[0]:9.3f}   {pinn_mae[1]:12.3f}")
    print(f"  plain network     : {base_mae[0]:9.3f}   {base_mae[1]:12.3f}")
    print("=" * 58)
    if pinn_mae.mean() < base_mae.mean():
        print("  The Arrhenius constraint improved generalisation.")
    else:
        print("  No improvement. Tune PHYSICS_WEIGHT, or the first-order")
        print("  assumption may be too crude for this reactor.")

    torch.save(model.state_dict(), str(config.PINN_WEIGHTS))


if __name__ == "__main__":
    main()
