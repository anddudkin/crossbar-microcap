"""How much of bl_star's bit-line decoupling benefit survives once the
array shares one peripheral routing resistor (r_col_end) before the sense
amp?

r_col_end is emitted regardless of star_columns and is documented
(crossbar/topology.py) to reintroduce cross-cell coupling in a column even
when star_columns has otherwise made every cell's bit-line wiring
independent — every cell in a column shares this one resistor. Existing
scripts only ever set r_col_end to a single fixed value; this sweeps
r_col_end as a *ratio* to R_col (not an absolute ohm value — see
run_error_landscape.py's docstring for why ratios, not absolute
magnitudes, are what varies the relative error in a linear resistive
network) across all four variant_configs, at a couple of representative
array sizes.

Usage:
    python3 examples/run_periphery_sweep.py
    python3 examples/run_periphery_sweep.py --sizes 8x8 32x32 128x128 \
        --ratios 0 0.1 0.3 1 3 10 30
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crossbar.topology import CrossbarConfig
from crossbar.compare import ideal_vmm, variant_configs, compute_errors, BACKENDS

# Same four hues run_error_landscape.py uses for these variants, so a
# variant's color means the same thing in every figure across both scripts.
VARIANT_COLORS = {
    "baseline": "#2a78d6",
    "wl_buffer_full": "#eb6834",
    "bl_star": "#1baf7a",
    "combined_full": "#eda100",
}


def parse_size(token: str) -> tuple[int, int]:
    rows, cols = token.lower().split("x")
    return int(rows), int(cols)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="+", default=["8x8", "64x64"], help="RxC tokens, e.g. 8x8 64x64")
    ap.add_argument("--r-line", type=float, default=10.0, help="ohms per WL/BL wire segment")
    ap.add_argument("--r-cell", type=float, default=1000.0, help="ohms per crosspoint cell")
    ap.add_argument("--v-in", type=float, default=0.7, help="volts on every row source")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0, 0.1, 0.3, 1, 3, 10],
                     help="r_col_end / r_col values to sweep")
    ap.add_argument("--dense-max-cells", type=int, default=2048,
                     help="use the dense MNA backend at/below this many cells (rows*cols), sparse above it")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "out")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sizes = [parse_size(s) for s in args.sizes]

    rows_out = []
    for rows, cols in sizes:
        backend_name = "dense" if rows * cols <= args.dense_max_cells else "sparse"
        compute = BACKENDS[backend_name]
        g = np.full((rows, cols), 1.0 / args.r_cell)
        v_in = np.full(rows, args.v_in)
        for ratio in args.ratios:
            base_cfg = CrossbarConfig(g=g, v_in=v_in, r_row=args.r_line, r_col=args.r_line,
                                       r_col_end=ratio * args.r_line)
            ideal = ideal_vmm(base_cfg)
            for label, cfg in variant_configs(base_cfg).items():
                currents = compute(cfg)
                rmse, max_abs, max_rel, mean_rel = compute_errors(ideal, currents)
                rows_out.append(dict(
                    rows=rows, cols=cols, r_col_end_ratio=ratio, r_col_end=base_cfg.r_col_end,
                    variant=label, rmse=rmse, max_abs_error=max_abs, max_rel_error=max_rel,
                    mean_rel_error=mean_rel, backend=backend_name,
                ))
        print(f"{rows}x{cols} ({backend_name}) done")

    csv_path = args.out / "periphery_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nSaved {csv_path}")

    fig, axes = plt.subplots(1, len(sizes), figsize=(6 * len(sizes), 5), squeeze=False)
    for ax, (rows, cols) in zip(axes[0], sizes):
        for variant, color in VARIANT_COLORS.items():
            xs = args.ratios
            ys = [max(row["mean_rel_error"] * 100, 1e-9) for row in rows_out
                  if row["rows"] == rows and row["cols"] == cols and row["variant"] == variant]
            ax.plot(xs, ys, marker="o", markersize=5, linewidth=2, color=color, label=variant)
        ax.set_yscale("log")
        ax.set_xlabel("r_col_end / R_col")
        ax.set_ylabel("mean relative error (%)")
        ax.set_title(f"{rows}x{cols}")
        ax.legend(fontsize=8)
    fig.suptitle("Effect of shared peripheral routing resistance (r_col_end)")
    fig.tight_layout()
    plot_path = args.out / "periphery_sweep.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
