"""End-to-end demo: build a resistive crossbar, compare IR-drop compensation
methods for VMM accuracy in ngspice, and render schematics + a summary plot.

Current experiment: uniform inputs and cell values (V_in=0.7V, R_cell=1kOhm)
so any variation across rows/columns comes purely from wire position, not
from data randomness — this isolates the IR-drop effect the compensation
methods are meant to fix. Compares baseline vs full WL buffering
(source_buffer + buffer_interval=1) vs per-cell BL sensing (star_columns)
vs both combined. (Partial/periodic WL buffering is a separate, later
question — see run_wl_buffer_interval_sweep.py.)

Usage:
    python3 examples/run_demo.py [--rows N] [--cols M] [--r-line OHMS]
                                  [--r-cell OHMS] [--v-in VOLTS] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crossbar.topology import CrossbarConfig
from crossbar.compare import ideal_vmm, compare_all, print_report
from crossbar.schematic import save_schematic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--r-line", type=float, default=10.0, help="ohms per WL/BL wire segment (interconnect)")
    ap.add_argument("--r-cell", type=float, default=1000.0, help="ohms per crosspoint cell")
    ap.add_argument("--v-in", type=float, default=0.7, help="volts on every row source")
    ap.add_argument("--buffer-r-out", type=float, default=0.0,
                     help="ohms of output resistance for every WL buffer (0 = ideal buffer)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "out")
    ap.add_argument("--schematic-size", type=int, default=4, help="NxN subset size used for illustration schematics")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    g = np.full((args.rows, args.cols), 1.0 / args.r_cell)
    v_in = np.full(args.rows, args.v_in)
    base_cfg = CrossbarConfig(
        g=g, v_in=v_in, r_row=args.r_line, r_col=args.r_line, buffer_r_out=args.buffer_r_out
    )

    print(f"Crossbar: {args.rows}x{args.cols}, R_line={args.r_line} ohm/segment, "
          f"R_cell={args.r_cell} ohm, V_in={args.v_in} V, buffer_r_out={args.buffer_r_out} ohm\n")

    ideal = ideal_vmm(base_cfg)
    results = compare_all(base_cfg)
    print_report(ideal, results)

    # --- CSV of results ---
    csv_path = args.out / "vmm_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "rmse_A", "max_abs_error_A", "mean_rel_error", "max_rel_error"])
        for r in results:
            writer.writerow([r.label, r.rmse, r.max_abs_error, r.mean_rel_error, r.max_rel_error])
    print(f"\nSaved {csv_path}")

    # --- bar chart of error metrics ---
    labels = [r.label for r in results]
    mean_rel = [r.mean_rel_error * 100 for r in results]
    max_rel = [r.max_rel_error * 100 for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.bar(labels, mean_rel, color="#4C72B0")
    ax1.set_ylabel("mean relative error (%)")
    ax1.set_title("Average column error")
    ax1.tick_params(axis="x", rotation=20)

    ax2.bar(labels, max_rel, color="#DD8452")
    ax2.set_ylabel("max relative error (%)")
    ax2.set_title("Worst-case column error")
    ax2.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    plot_path = args.out / "error_comparison.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved {plot_path}")

    # --- illustrative schematics (small NxN subset, full array is too dense to draw) ---
    k = min(args.schematic_size, args.rows, args.cols)
    illus_cfg = CrossbarConfig(
        g=g[:k, :k], v_in=v_in[:k], r_row=args.r_line, r_col=args.r_line, buffer_r_out=args.buffer_r_out
    )
    variants = {
        "baseline": replace(illus_cfg, buffer_interval=0, source_buffer=False, star_columns=False),
        "wl_buffer_full": replace(illus_cfg, buffer_interval=1, source_buffer=True, star_columns=False),
        "bl_star": replace(illus_cfg, buffer_interval=0, source_buffer=False, star_columns=True),
        "combined_full": replace(illus_cfg, buffer_interval=1, source_buffer=True, star_columns=True),
    }
    sch_dir = args.out / "schematics"
    sch_dir.mkdir(exist_ok=True)
    for name, cfg in variants.items():
        save_schematic(cfg, str(sch_dir / f"{name}.svg"), str(sch_dir / f"{name}.png"))
    print(f"Saved schematics to {sch_dir}")


if __name__ == "__main__":
    main()
