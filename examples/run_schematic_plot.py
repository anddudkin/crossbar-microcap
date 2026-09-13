"""Standalone schematic rendering with parameters set directly in code (no
CLI flags). Unlike run_demo.py this does not run ngspice/compare anything --
it only draws the crossbar schematic(s) for whichever compensation flags you
set below, at whatever size you set.

Edit the block below, then just run:

    python3 examples/run_schematic_plot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from crossbar.topology import CrossbarConfig
from crossbar.schematic import save_schematic

# ---------------------------------------------------------------------------
# Parameters -- edit these directly.
# ---------------------------------------------------------------------------

N_ROWS = 4
N_COLS = 4

R_CELL = 1000.0  # ohms, uniform crosspoint cell resistance
V_IN = 0.7  # volts, uniform row input voltage

R_ROW = 10.0  # word-line (row) segment resistance, ohms
R_COL = 10.0  # bit-line (column) segment resistance, ohms

# Compensation flags -- see CrossbarConfig in crossbar/topology.py.
BUFFER_INTERVAL = 0  # 0 = no periodic WL buffering; N = buffer every Nth crosspoint
SOURCE_BUFFER = False  # True + BUFFER_INTERVAL=1 = full WL compensation
BUFFER_R_OUT = 0.0  # ohms, output resistance of every buffer (0 = ideal)
STAR_COLUMNS = False  # True = dedicated per-cell column wiring (BL compensation)
R_COL_END = 0.0  # ohms, shared bit-line-end resistance (0 = none)

OUT_DIR = Path(__file__).resolve().parent.parent / "out" / "schematics"
OUT_NAME = "custom"  # file stem -> custom.svg / custom.png
TITLE = None  # None = auto-generate from the config

# ---------------------------------------------------------------------------


def main() -> None:
    g = np.full((N_ROWS, N_COLS), 1.0 / R_CELL)
    v_in = np.full(N_ROWS, V_IN)
    cfg = CrossbarConfig(
        g=g, v_in=v_in, r_row=R_ROW, r_col=R_COL,
        buffer_interval=BUFFER_INTERVAL, source_buffer=SOURCE_BUFFER,
        buffer_r_out=BUFFER_R_OUT, star_columns=STAR_COLUMNS, r_col_end=R_COL_END,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUT_DIR / f"{OUT_NAME}.svg"
    png_path = OUT_DIR / f"{OUT_NAME}.png"
    save_schematic(cfg, str(svg_path), str(png_path), title=TITLE)

    print(f"Crossbar: {cfg.variant_label()} ({N_ROWS}x{N_COLS}), R_row={R_ROW} ohm, "
          f"R_col={R_COL} ohm, buffer_r_out={BUFFER_R_OUT} ohm, r_col_end={R_COL_END} ohm")
    print(f"Saved {svg_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
