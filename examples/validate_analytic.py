"""Cross-validate the ngspice simulation against an independent analytical
solution of the same resistor network (crossbar.analytic, direct Modified
Nodal Analysis via numpy — no SPICE involved).

Two things are checked:
1. ngspice vs MNA agree to numerical precision on all four compensation
   variants (baseline / wl_buffer_full / bl_star / combined_full), with and
   without a non-ideal buffer_r_out — this is a correctness check on the
   SPICE netlist itself, not an approximation, since the network is linear
   and both methods solve the same equations exactly.
2. For combined_full specifically, both agree with the closed-form formula
   I_j = sum_i V_i / (R_cell_ij + R_col * (n - i)) — the exact result once
   WL IR-drop is fully cancelled (source_buffer + buffer_interval=1) and BL
   cross-coupling is removed (star_columns), leaving only each cell's own
   (uncoupled) voltage divider against its dedicated column resistance.

Usage:
    python3 examples/validate_analytic.py [--rows N] [--cols M]
                                           [--r-line OHMS] [--r-cell OHMS]
                                           [--v-in VOLTS] [--seed S]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from crossbar.topology import CrossbarConfig
from crossbar.compare import run_variant
from crossbar.analytic import solve_analytic


def closed_form_combined_full(cfg: CrossbarConfig) -> np.ndarray:
    n = cfg.n_rows
    dist = (n - np.arange(n))[:, None]  # (n, 1), distance-to-bottom per row
    r_cell = 1.0 / cfg.g  # (n, m)
    return np.sum(cfg.v_in[:, None] / (r_cell + cfg.r_col * dist), axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--r-line", type=float, default=10.0)
    ap.add_argument("--r-cell", type=float, default=1000.0)
    ap.add_argument("--v-in", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0, help="use a random (non-uniform) array instead of uniform")
    ap.add_argument("--random", action="store_true", help="randomize g/v_in instead of using uniform values")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.random:
        g = rng.uniform(0.2e-3, 1.0 / args.r_cell, size=(args.rows, args.cols))
        v_in = rng.uniform(0.2, args.v_in, size=args.rows)
    else:
        g = np.full((args.rows, args.cols), 1.0 / args.r_cell)
        v_in = np.full(args.rows, args.v_in)

    base = CrossbarConfig(g=g, v_in=v_in, r_row=args.r_line, r_col=args.r_line)

    variants = {
        "baseline": replace(base, buffer_interval=0, source_buffer=False, star_columns=False),
        "wl_buffer_full": replace(base, buffer_interval=1, source_buffer=True, star_columns=False),
        "bl_star": replace(base, buffer_interval=0, source_buffer=False, star_columns=True),
        "combined_full": replace(base, buffer_interval=1, source_buffer=True, star_columns=True),
        "wl_buffer_full (buffer_r_out=5)": replace(
            base, buffer_interval=1, source_buffer=True, star_columns=False, buffer_r_out=5.0
        ),
        "combined_full (buffer_r_out=5)": replace(
            base, buffer_interval=1, source_buffer=True, star_columns=True, buffer_r_out=5.0
        ),
    }

    print(f"Crossbar: {args.rows}x{args.cols}, R_line={args.r_line} ohm, R_cell base={args.r_cell} ohm, "
          f"{'random' if args.random else 'uniform'} array\n")

    header = f"{'variant':<34}{'max |ngspice - MNA| (A)':>26}"
    print(header)
    print("-" * len(header))
    for name, cfg in variants.items():
        ng = run_variant(cfg)
        an = solve_analytic(cfg)
        print(f"{name:<34}{np.max(np.abs(ng - an)):>26.3e}")

    print("\ncombined_full vs closed-form I_j = sum_i V_i/(R_cell_ij + R_col*(n-i)):")
    cf_cfg = variants["combined_full"]
    closed = closed_form_combined_full(cf_cfg)
    an = solve_analytic(cf_cfg)
    ng = run_variant(cf_cfg)
    print(f"  closed-form : {np.array2string(closed, precision=8)}")
    print(f"  MNA         : {np.array2string(an, precision=8)}")
    print(f"  ngspice     : {np.array2string(ng, precision=8)}")
    print(f"  max |closed-form - ngspice| = {np.max(np.abs(closed - ng)):.3e} A")


if __name__ == "__main__":
    main()
