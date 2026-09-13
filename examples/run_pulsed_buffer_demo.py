"""Comparator/inverter-repeater buffer demo — a second, fully separate
buffer model for *fixed-amplitude* (pulsed/spiking) row signalling, kept out
of compare.py's four analog-VMM variants entirely (see
CrossbarConfig.comparator_buffer).

Where the project's main buffer (source_buffer/buffer_interval) is an ideal
linear unity-gain VCVS that reproduces whatever V_i is, this variant
(comparator_buffer=True) instead mirrors the original Micro-Cap repeater
(reference/microcap/base_r_line_10_WL_invertor.cir's `.SUBCKT INV`: a
threshold comparator, not a linear amplifier) — appropriate when every row
signal really is one of two fixed logic levels (buffer_v_low/buffer_v_high)
rather than an arbitrary analog value, as in a spiking/pulsed network.

Because every buffer here is referenced to the row's own *ideal, undropped*
source node, the comparator's HIGH/LOW decision is a known Python float at
netlist-build time (see CrossbarConfig.buffer_output_level) — so it's
stamped as an ordinary fixed DC source, not a genuine nonlinear element.
That keeps it strictly linear, so ngspice / dense MNA / sparse MNA all still
agree to floating-point precision, exactly like every other variant in this
project; that three-way agreement is exactly what this script checks.

Note on scoring: the "ideal" target for the comparator variant is not
v_in @ g (that's the ideal for a *linear* buffer, which reproduces v_in
itself) but buffer_output_level(i) @ g — the perfectly-regenerated rail
levels a repeater is actually trying to deliver. Snapping V_i to a rail is
the comparator's intended behaviour, not an IR-drop error, so scoring it
against raw v_in would conflate the two.

Usage:
    python3 examples/run_pulsed_buffer_demo.py --rows 8 --cols 8 \
        --r-line 10 --r-cell 1000 --v-in 0.1 0.7 --schematic
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from crossbar.topology import CrossbarConfig
from crossbar.compare import ideal_vmm, compute_errors, run_variant
from crossbar.analytic import solve_analytic, solve_analytic_sparse
from crossbar.schematic import save_schematic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--r-line", type=float, default=10.0, help="ohms per WL/BL wire segment")
    ap.add_argument("--r-cell", type=float, default=1000.0, help="ohms per crosspoint cell")
    ap.add_argument("--v-in", type=float, nargs="+", default=[0.7],
                     help="row input voltage(s), volts; cycled across rows if fewer than "
                          "--rows values are given (e.g. --v-in 0.1 0.7 alternates a "
                          "below-threshold and an above-threshold row)")
    ap.add_argument("--buffer-r-out", type=float, default=0.0,
                     help="ohms of output resistance for every buffer (0 = ideal)")
    ap.add_argument("--buffer-threshold", type=float, default=0.35, help="comparator decision point, volts")
    ap.add_argument("--buffer-v-low", type=float, default=0.1, help="comparator LOW rail, volts")
    ap.add_argument("--buffer-v-high", type=float, default=0.7, help="comparator HIGH rail, volts")
    ap.add_argument("--skip-ngspice", action="store_true",
                     help="skip ngspice and only cross-check the dense/sparse analytic backends")
    ap.add_argument("--schematic", action="store_true",
                     help="also render a small schematic showing the comparator/inverter symbol")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "out")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    g = np.full((args.rows, args.cols), 1.0 / args.r_cell)
    v_in = np.array([args.v_in[i % len(args.v_in)] for i in range(args.rows)])

    base_cfg = CrossbarConfig(
        g=g, v_in=v_in, r_row=args.r_line, r_col=args.r_line, buffer_r_out=args.buffer_r_out,
        source_buffer=True, buffer_interval=1,
        buffer_threshold=args.buffer_threshold, buffer_v_low=args.buffer_v_low, buffer_v_high=args.buffer_v_high,
    )
    cfg_linear = replace(base_cfg, comparator_buffer=False)
    cfg_pulsed = replace(base_cfg, comparator_buffer=True)

    print(f"Crossbar: {args.rows}x{args.cols}, R_line={args.r_line} ohm, R_cell={args.r_cell} ohm, "
          f"buffer_r_out={args.buffer_r_out} ohm")
    print(f"Row inputs V_in: {np.array2string(v_in, precision=3)} V")
    print(f"Comparator: threshold={args.buffer_threshold} V, "
          f"LOW={args.buffer_v_low} V, HIGH={args.buffer_v_high} V\n")

    def backend_currents(cfg: CrossbarConfig) -> dict[str, np.ndarray]:
        currents = {"dense": solve_analytic(cfg), "sparse": solve_analytic_sparse(cfg)}
        if not args.skip_ngspice:
            currents["ngspice"] = run_variant(cfg)
        return currents

    # --- linear (existing ideal unity-gain buffer): scored against v_in @ g ---
    print(f"{cfg_linear.variant_label()} (linear unity-gain buffer)")
    linear_currents = backend_currents(cfg_linear)
    ideal_linear = ideal_vmm(cfg_linear)
    rmse, max_abs, max_rel, mean_rel = compute_errors(ideal_linear, linear_currents["dense"])
    print(f"  ideal (v_in @ g)      : {np.array2string(ideal_linear, precision=6)}")
    print(f"  dense                 : {np.array2string(linear_currents['dense'], precision=6)}")
    print(f"  RMSE={rmse:.3e} A, mean rel err={mean_rel:.2%}, max rel err={max_rel:.2%}")

    # --- pulsed (comparator/inverter repeater): scored against the ---
    # --- perfectly-regenerated rail levels, not raw v_in (see module docstring) ---
    print(f"\n{cfg_pulsed.variant_label()} (comparator/inverter repeater)")
    pulsed_currents = backend_currents(cfg_pulsed)
    levels = np.array([cfg_pulsed.buffer_output_level(i) for i in range(cfg_pulsed.n_rows)])
    ideal_pulsed = levels @ g
    rmse, max_abs, max_rel, mean_rel = compute_errors(ideal_pulsed, pulsed_currents["dense"])
    print(f"  regenerated rail levels per row: {np.array2string(levels, precision=3)} V")
    print(f"  ideal (levels @ g)    : {np.array2string(ideal_pulsed, precision=6)}")
    print(f"  dense                 : {np.array2string(pulsed_currents['dense'], precision=6)}")
    print(f"  RMSE={rmse:.3e} A, mean rel err={mean_rel:.2%}, max rel err={max_rel:.2%}")

    print("\nCross-backend agreement for the comparator variant (should be at floating-point precision):")
    print(f"  max |dense - sparse|  = {np.max(np.abs(pulsed_currents['dense'] - pulsed_currents['sparse'])):.3e} A")
    if not args.skip_ngspice:
        print(f"  max |dense - ngspice| = {np.max(np.abs(pulsed_currents['dense'] - pulsed_currents['ngspice'])):.3e} A")

    if args.schematic:
        out_dir = args.out / "schematics"
        out_dir.mkdir(parents=True, exist_ok=True)
        demo_rows, demo_cols = min(args.rows, 4), min(args.cols, 4)
        demo_cfg = replace(cfg_pulsed, g=g[:demo_rows, :demo_cols], v_in=v_in[:demo_rows])
        svg_path, png_path = out_dir / "pulsed_buffer_demo.svg", out_dir / "pulsed_buffer_demo.png"
        save_schematic(demo_cfg, str(svg_path), str(png_path))
        print(f"\nSaved {svg_path}\nSaved {png_path}")


if __name__ == "__main__":
    main()
