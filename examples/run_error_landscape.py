"""Main paper experiment: how does IR-drop error depend jointly on array
size and the R_line/R_cell resistance ratio, across no/partial/full
compensation?

Existing scripts spot-check one array size and one R_line/R_cell pair at a
time (run_demo.py, run_compare.py). This sweeps both axes on a grid and
evaluates every compensation level at each point in one pass, using the
dense/sparse MNA backends from crossbar.analytic (see BACKENDS in
crossbar/compare.py) instead of ngspice so it scales to large arrays.

Key physical fact this sweep relies on: the crossbar is a *linear*
resistive network, so scaling every resistance (R_row, R_col, R_cell,
buffer_r_out) by the same factor leaves every relative-error metric
exactly unchanged — only the *ratio* between them matters, not their
absolute magnitude. That's why the swept quantity is R_line/R_cell (with
R_cell held fixed) rather than R_line and R_cell independently, and why
the two non-ideal-buffer variants below express buffer_r_out as a
fraction of R_row rather than a fixed ohm value. (--verify-invariance
checks this property directly: two grid points sharing a ratio but
different R_cell must produce identical errors.)

Eight variants are evaluated at every grid point, spanning none -> partial
-> full compensation:
    baseline              (none)
    wl_partial_2/4        (partial: repeater every 2nd/4th crosspoint)
    wl_full_nonideal_lo/hi(partial: full buffering, but buffer_r_out is
                            10%/50% of R_row instead of ideal)
    wl_buffer_full        (full: row IR-drop fully cancelled)
    bl_star               (full: no shared bit-line coupling)
    combined_full         (full: both)

Usage:
    python3 examples/run_error_landscape.py
    python3 examples/run_error_landscape.py --sizes 4x4 16x16 64x64 \
        --ratios 0.001 0.01 0.1 --verify-ngspice
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from crossbar.topology import CrossbarConfig
from crossbar.compare import ideal_vmm, compute_errors, BACKENDS

# Fixed hue per variant, held constant across every figure this script (and
# run_periphery_sweep.py, for the four it shares) produces — color follows
# the entity, never plot-local ordering. Slots 1-4 match variant_configs'
# four canonical variants so figures from both scripts stay consistent.
VARIANT_COLORS = {
    "baseline": "#2a78d6",
    "wl_buffer_full": "#eb6834",
    "bl_star": "#1baf7a",
    "combined_full": "#eda100",
    "wl_partial_2": "#e87ba4",
    "wl_partial_4": "#008300",
    "wl_full_nonideal_lo": "#4a3aa7",
    "wl_full_nonideal_hi": "#e34948",
}
FAMILY_OF = {
    "baseline": "none",
    "wl_buffer_full": "full",
    "bl_star": "full",
    "combined_full": "full",
    "wl_partial_2": "partial",
    "wl_partial_4": "partial",
    "wl_full_nonideal_lo": "partial",
    "wl_full_nonideal_hi": "partial",
}


def parse_size(token: str) -> tuple[int, int]:
    rows, cols = token.lower().split("x")
    return int(rows), int(cols)


def build_variants(base_cfg: CrossbarConfig) -> dict[str, CrossbarConfig]:
    """The eight none/partial/full variants for this grid point. buffer_r_out
    is expressed as a fraction of base_cfg.r_row (not a fixed ohm value) so
    it sweeps meaningfully together with the R_line/R_cell ratio axis — see
    module docstring."""
    r_out = base_cfg.r_row
    return {
        "baseline": replace(base_cfg, buffer_interval=0, source_buffer=False, star_columns=False, buffer_r_out=0.0),
        "wl_partial_2": replace(base_cfg, buffer_interval=2, source_buffer=False, star_columns=False, buffer_r_out=0.0),
        "wl_partial_4": replace(base_cfg, buffer_interval=4, source_buffer=False, star_columns=False, buffer_r_out=0.0),
        "wl_full_nonideal_lo": replace(base_cfg, buffer_interval=1, source_buffer=True, star_columns=False, buffer_r_out=0.1 * r_out),
        "wl_full_nonideal_hi": replace(base_cfg, buffer_interval=1, source_buffer=True, star_columns=False, buffer_r_out=0.5 * r_out),
        "wl_buffer_full": replace(base_cfg, buffer_interval=1, source_buffer=True, star_columns=False, buffer_r_out=0.0),
        "bl_star": replace(base_cfg, buffer_interval=0, source_buffer=False, star_columns=True, buffer_r_out=0.0),
        "combined_full": replace(base_cfg, buffer_interval=1, source_buffer=True, star_columns=True, buffer_r_out=0.0),
    }


def make_base_cfg(rows: int, cols: int, r_cell: float, ratio: float, v_in: float) -> CrossbarConfig:
    r_line = ratio * r_cell
    g = np.full((rows, cols), 1.0 / r_cell)
    v = np.full(rows, v_in)
    return CrossbarConfig(g=g, v_in=v, r_row=r_line, r_col=r_line)


def run_grid(sizes, ratios, r_cell, v_in, dense_max_cells) -> list[dict]:
    rows_out = []
    total = len(sizes) * len(ratios)
    done = 0
    for rows, cols in sizes:
        backend_name = "dense" if rows * cols <= dense_max_cells else "sparse"
        compute = BACKENDS[backend_name]
        for ratio in ratios:
            base_cfg = make_base_cfg(rows, cols, r_cell, ratio, v_in)
            ideal = ideal_vmm(base_cfg)
            for label, cfg in build_variants(base_cfg).items():
                t0 = time.perf_counter()
                currents = compute(cfg)
                elapsed = time.perf_counter() - t0
                rmse, max_abs, max_rel, mean_rel = compute_errors(ideal, currents)
                rows_out.append(dict(
                    rows=rows, cols=cols, ratio=ratio, r_row=base_cfg.r_row, r_cell=r_cell,
                    variant=label, family=FAMILY_OF[label], rmse=rmse, max_abs_error=max_abs,
                    max_rel_error=max_rel, mean_rel_error=mean_rel, backend=backend_name, elapsed_s=elapsed,
                ))
            done += 1
            print(f"[{done}/{total}] {rows}x{cols}, ratio={ratio:g} ({backend_name}) done")
    return rows_out


def verify_invariance(sizes, ratios, r_cell, v_in, dense_max_cells) -> None:
    """Free correctness check: two grid points sharing a ratio but a
    different absolute R_cell must give identical relative errors, since
    the network is linear (see module docstring)."""
    rows, cols = sizes[0]
    ratio = ratios[0]
    a = run_grid([(rows, cols)], [ratio], r_cell, v_in, dense_max_cells)
    b = run_grid([(rows, cols)], [ratio], r_cell * 7.0, v_in, dense_max_cells)
    worst = max(abs(x["mean_rel_error"] - y["mean_rel_error"]) for x, y in zip(a, b))
    print(f"\nInvariance check ({rows}x{cols}, ratio={ratio:g}): "
          f"R_cell={r_cell:g} vs {r_cell * 7.0:g} ohm -> "
          f"max |mean_rel_error diff| = {worst:.3e} (should be ~0)\n")


def verify_ngspice(sizes, ratios, r_cell, v_in) -> None:
    """Sanity spot-check the smallest grid point's dense/sparse result
    against ngspice, for baseline and combined_full — the full
    ngspice/dense/sparse cross-validation already lives in
    validate_analytic.py, this is just a one-point confidence check."""
    rows, cols = sizes[0]
    base_cfg = make_base_cfg(rows, cols, r_cell, ratios[0], v_in)
    ideal = ideal_vmm(base_cfg)
    backend_name = "dense" if rows * cols <= 2048 else "sparse"
    for label, cfg in (("baseline", build_variants(base_cfg)["baseline"]),
                        ("combined_full", build_variants(base_cfg)["combined_full"])):
        analytic = BACKENDS[backend_name](cfg)
        spice = BACKENDS["ngspice"](cfg)
        diff = float(np.max(np.abs(analytic - spice)))
        print(f"verify-ngspice [{label}] {backend_name} vs ngspice: max |diff| = {diff:.3e} A")


def plot_heatmaps(rows_out, sizes, ratios, out_path) -> None:
    variants = list(VARIANT_COLORS)
    size_labels = [f"{r}x{c}" for r, c in sizes]
    grid = {v: np.full((len(sizes), len(ratios)), np.nan) for v in variants}
    for row in rows_out:
        i = sizes.index((row["rows"], row["cols"]))
        j = ratios.index(row["ratio"])
        grid[row["variant"]][i, j] = max(row["mean_rel_error"] * 100, 1e-6)

    vmin = min(np.nanmin(grid[v]) for v in variants)
    vmax = max(np.nanmax(grid[v]) for v in variants)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    im = None
    for ax, v in zip(axes.flat, variants):
        im = ax.imshow(grid[v], aspect="auto", cmap="Blues", norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_title(v, fontsize=10, color="#0b0b0b")
        ax.set_xticks(range(len(ratios)))
        ax.set_xticklabels([f"{r:g}" for r in ratios], rotation=45, fontsize=7)
        ax.set_yticks(range(len(size_labels)))
        ax.set_yticklabels(size_labels, fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.8, label="mean relative error (%)")
    fig.suptitle("Mean relative error vs array size and R_line/R_cell ratio")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_vs_size(rows_out, sizes, ratio_slice, out_path) -> None:
    square = [(r, c) for r, c in sizes if r == c]
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant, color in VARIANT_COLORS.items():
        xs, ys = [], []
        for r, c in square:
            match = [row for row in rows_out if row["rows"] == r and row["cols"] == c
                     and row["ratio"] == ratio_slice and row["variant"] == variant]
            if match:
                xs.append(r)
                ys.append(max(match[0]["mean_rel_error"] * 100, 1e-9))
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=2, color=color, label=variant)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("array size N (NxN)")
    ax.set_ylabel("mean relative error (%)")
    ax.set_title(f"Error vs array size (R_line/R_cell = {ratio_slice:g})")
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_vs_ratio(rows_out, ratios, size_slice, out_path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant, color in VARIANT_COLORS.items():
        xs, ys = [], []
        for ratio in ratios:
            match = [row for row in rows_out if row["rows"] == size_slice[0] and row["cols"] == size_slice[1]
                     and row["ratio"] == ratio and row["variant"] == variant]
            if match:
                xs.append(ratio)
                ys.append(max(match[0]["mean_rel_error"] * 100, 1e-9))
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=2, color=color, label=variant)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("R_line / R_cell")
    ax.set_ylabel("mean relative error (%)")
    ax.set_title(f"Error vs resistance ratio ({size_slice[0]}x{size_slice[1]})")
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="+", default=["4x4", "8x8", "16x16", "32x32", "64x64", "128x128", "256x256"],
                     help="RxC tokens, e.g. 8x8 16x32")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.001, 0.003, 0.01, 0.03, 0.1],
                     help="R_line / R_cell values to sweep (R_cell held fixed)")
    ap.add_argument("--r-cell", type=float, default=1000.0, help="ohms per crosspoint cell (fixed; only the ratio matters)")
    ap.add_argument("--v-in", type=float, default=0.7, help="volts on every row source")
    ap.add_argument("--dense-max-cells", type=int, default=2048,
                     help="use the dense MNA backend at/below this many cells (rows*cols), sparse above it")
    ap.add_argument("--verify-ngspice", action="store_true",
                     help="spot-check the smallest grid point's backend against ngspice")
    ap.add_argument("--verify-invariance", action="store_true",
                     help="confirm relative error only depends on the ratio, not absolute R_cell")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "out")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sizes = [parse_size(s) for s in args.sizes]
    ratios = args.ratios

    if args.verify_ngspice:
        verify_ngspice(sizes, ratios, args.r_cell, args.v_in)
    if args.verify_invariance:
        verify_invariance(sizes, ratios, args.r_cell, args.v_in, args.dense_max_cells)

    rows_out = run_grid(sizes, ratios, args.r_cell, args.v_in, args.dense_max_cells)

    csv_path = args.out / "error_landscape.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nSaved {csv_path}")

    plot_heatmaps(rows_out, sizes, ratios, args.out / "error_landscape_heatmap.png")
    plot_vs_size(rows_out, sizes, ratios[len(ratios) // 2], args.out / "error_landscape_vs_size.png")
    plot_vs_ratio(rows_out, ratios, sizes[len(sizes) // 2], args.out / "error_landscape_vs_ratio.png")


if __name__ == "__main__":
    main()
