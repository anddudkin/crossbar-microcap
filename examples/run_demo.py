"""End-to-end demo: build a resistive crossbar, compare IR-drop compensation
methods for VMM accuracy in ngspice, and render schematics + a summary plot.

Usage:
    python3 examples/run_demo.py [--rows N] [--cols M] [--r-line OHMS]
                                  [--buffer-interval K] [--seed S] [--out DIR]
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
from crossbar.compare import ideal_vmm, compare_all, print_report
from crossbar.schematic import save_schematic
from dataclasses import replace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--r-line", type=float, default=10.0, help="ohms per WL/BL segment")
    ap.add_argument("--buffer-interval", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "out")
    ap.add_argument("--schematic-size", type=int, default=4, help="NxN subset size used for illustration schematics")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    g = rng.uniform(0.1e-3, 1e-3, size=(args.rows, args.cols))
    v_in = rng.uniform(0.1, 1.0, size=args.rows)
    base_cfg = CrossbarConfig(g=g, v_in=v_in, r_row=args.r_line, r_col=args.r_line)

    print(f"Crossbar: {args.rows}x{args.cols}, R_line={args.r_line} ohm/segment, "
          f"buffer_interval={args.buffer_interval}\n")

    ideal = ideal_vmm(base_cfg)
    results = compare_all(base_cfg, buffer_interval=args.buffer_interval)
    print_report(ideal, results)

    # --- CSV of results ---
    csv_path = args.out / "vmm_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "rmse_A", "max_abs_error_A", "max_rel_error"])
        for r in results:
            writer.writerow([r.label, r.rmse, r.max_abs_error, r.max_rel_error])
    print(f"\nSaved {csv_path}")

    # --- bar chart of error metrics ---
    labels = [r.label for r in results]
    rmse = [r.rmse for r in results]
    max_rel = [r.max_rel_error * 100 for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.bar(labels, rmse, color="#4C72B0")
    ax1.set_ylabel("RMSE (A)")
    ax1.set_title("VMM output RMSE vs ideal")
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
        g=g[:k, :k], v_in=v_in[:k], r_row=args.r_line, r_col=args.r_line
    )
    variants = {
        "baseline": replace(illus_cfg, buffer_interval=0, star_columns=False),
        "wl_buffer": replace(illus_cfg, buffer_interval=args.buffer_interval, star_columns=False),
        "bl_star": replace(illus_cfg, buffer_interval=0, star_columns=True),
        "combined": replace(illus_cfg, buffer_interval=args.buffer_interval, star_columns=True),
    }
    sch_dir = args.out / "schematics"
    sch_dir.mkdir(exist_ok=True)
    for name, cfg in variants.items():
        save_schematic(cfg, str(sch_dir / f"{name}.svg"), str(sch_dir / f"{name}.png"))
    print(f"Saved schematics to {sch_dir}")


if __name__ == "__main__":
    main()
