"""Research question set aside for later: how does *partial/periodic* WL
buffering (a repeater only every N-th crosspoint, no source buffer) trade
off against the *full* per-cell compensation used in run_demo.py?

This keeps the original buffer_interval-sweep idea (one repeater every N
cells, as in reference/microcap/base_r_line_10_WL_invertor.cir) in its own
place so it doesn't get lost, without mixing it into the main experiment
in run_demo.py (which now always uses full compensation:
source_buffer=True, buffer_interval=1).

Usage:
    python3 examples/run_wl_buffer_interval_sweep.py --rows 8 --cols 8 \
        --r-line 10 --r-cell 1000 --v-in 0.7 --intervals 1 2 3 4 8
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from crossbar.topology import CrossbarConfig
from crossbar.compare import ideal_vmm, evaluate_variant


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--r-line", type=float, default=10.0, help="ohms per WL/BL segment")
    ap.add_argument("--r-cell", type=float, default=1000.0, help="ohms per crosspoint cell")
    ap.add_argument("--v-in", type=float, default=0.7, help="volts on every row source")
    ap.add_argument("--intervals", type=int, nargs="+", default=[1, 2, 3, 4, 8],
                     help="buffer_interval values to sweep (no source buffer)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "out")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    g = np.full((args.rows, args.cols), 1.0 / args.r_cell)
    v_in = np.full(args.rows, args.v_in)
    base_cfg = CrossbarConfig(g=g, v_in=v_in, r_row=args.r_line, r_col=args.r_line)

    ideal = ideal_vmm(base_cfg)
    print(f"Crossbar: {args.rows}x{args.cols}, R_line={args.r_line} ohm, "
          f"R_cell={args.r_cell} ohm, V_in={args.v_in} V\n")
    print(f"Ideal (R_line=0) column currents: {np.array2string(ideal, precision=6)}\n")

    header = (
        f"{'buffer_interval':<16}{'RMSE (A)':>14}{'max |err| (A)':>16}"
        f"{'mean rel err':>14}{'max rel err':>14}"
    )
    print(header)
    print("-" * len(header))

    rows_out = []
    for k in args.intervals:
        cfg = replace(base_cfg, buffer_interval=k, source_buffer=False, star_columns=False)
        result = evaluate_variant(f"interval_{k}", cfg, ideal)
        print(
            f"{k:<16}{result.rmse:>14.3e}{result.max_abs_error:>16.3e}"
            f"{result.mean_rel_error:>14.2%}{result.max_rel_error:>14.2%}"
        )
        rows_out.append(
            (k, result.rmse, result.max_abs_error, result.mean_rel_error, result.max_rel_error)
        )

    csv_path = args.out / "wl_buffer_interval_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["buffer_interval", "rmse_A", "max_abs_error_A", "mean_rel_error", "max_rel_error"])
        writer.writerows(rows_out)
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
