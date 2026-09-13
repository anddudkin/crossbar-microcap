"""Ad-hoc comparison tool: set array size, wire/cell resistance and input
voltage on the command line, and choose which compensation variant(s) to
actually run — all four (default) or any subset.

Unlike run_demo.py (which always runs all four variants with its own
fixed "main experiment" framing and also renders schematics), this is a
lighter, general-purpose knob-turning tool for quick what-if numbers:
different resistances, a different array size, or just one or two
variants instead of all four. It also lets you pick the solver backend
(ngspice, or one of the two MNA backends from crossbar.analytic — see
--backend), which matters for large arrays where ngspice/dense MNA become
impractical (see examples/validate_analytic.py).

Examples:
    # all four variants, custom resistances
    python3 examples/run_compare.py --r-line 1 --r-cell 10000 --v-in 0.7

    # only baseline and the fully combined method
    python3 examples/run_compare.py --variants baseline combined_full

    # large array: skip ngspice/dense (both impractical there), use sparse MNA
    python3 examples/run_compare.py --rows 128 --cols 128 --backend sparse

    # save a CSV of the results
    python3 examples/run_compare.py --csv out/my_comparison.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from crossbar.topology import CrossbarConfig
from crossbar.compare import ideal_vmm, variant_configs, compute_errors, BACKENDS

ALL_VARIANTS = ["baseline", "wl_buffer_full", "bl_star", "combined_full"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--r-line", type=float, default=10.0, help="ohms per WL/BL wire segment (interconnect)")
    ap.add_argument("--r-cell", type=float, default=1000.0, help="ohms per crosspoint cell")
    ap.add_argument("--v-in", type=float, default=0.7, help="volts on every row source")
    ap.add_argument("--buffer-r-out", type=float, default=0.0,
                     help="ohms of output resistance for every WL buffer (0 = ideal buffer)")
    ap.add_argument("--r-col-end", type=float, default=0.0,
                     help="ohms of shared bit-line-end resistance before the sense amp (0 = none)")
    ap.add_argument("--random", action="store_true", help="randomize g/v_in instead of using uniform values")
    ap.add_argument("--seed", type=int, default=0, help="seed for --random")
    ap.add_argument("--variants", nargs="+", choices=ALL_VARIANTS, default=ALL_VARIANTS,
                     help="which compensation variant(s) to run (default: all four)")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="ngspice",
                     help="ngspice (default) or one of the MNA solvers from crossbar.analytic — "
                          "'dense' and especially 'sparse' matter for arrays too large for "
                          "ngspice/dense to handle in reasonable time (see validate_analytic.py)")
    ap.add_argument("--csv", type=Path, default=None, help="optional path to save results as CSV")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.random:
        g = rng.uniform(0.2e-3, 1.0 / args.r_cell, size=(args.rows, args.cols))
        v_in = rng.uniform(0.2, args.v_in, size=args.rows)
    else:
        g = np.full((args.rows, args.cols), 1.0 / args.r_cell)
        v_in = np.full(args.rows, args.v_in)

    base_cfg = CrossbarConfig(
        g=g, v_in=v_in, r_row=args.r_line, r_col=args.r_line,
        buffer_r_out=args.buffer_r_out, r_col_end=args.r_col_end,
    )

    print(f"Crossbar: {args.rows}x{args.cols}, R_line={args.r_line} ohm/segment, "
          f"R_cell={args.r_cell} ohm, V_in={args.v_in} V, buffer_r_out={args.buffer_r_out} ohm, "
          f"r_col_end={args.r_col_end} ohm, backend={args.backend}, "
          f"{'random' if args.random else 'uniform'} array\n")

    ideal = ideal_vmm(base_cfg)
    print(f"Ideal (R_line=0) column currents: {np.array2string(ideal, precision=6)}\n")

    compute = BACKENDS[args.backend]
    configs = variant_configs(base_cfg)

    header = (
        f"{'variant':<16}{'RMSE (A)':>14}{'max |err| (A)':>16}"
        f"{'mean rel err':>14}{'max rel err':>14}"
    )
    print(header)
    print("-" * len(header))

    rows_out = []
    for name in args.variants:
        currents = compute(configs[name])
        rmse, max_abs, max_rel, mean_rel = compute_errors(ideal, currents)
        print(f"{name:<16}{rmse:>14.3e}{max_abs:>16.3e}{mean_rel:>14.2%}{max_rel:>14.2%}")
        rows_out.append((name, rmse, max_abs, mean_rel, max_rel))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["variant", "rmse_A", "max_abs_error_A", "mean_rel_error", "max_rel_error"])
            writer.writerows(rows_out)
        print(f"\nSaved {args.csv}")


if __name__ == "__main__":
    main()
