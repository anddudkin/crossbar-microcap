"""Custom crossbar run with parameters set directly in code (no CLI flags).

Edit the block below to set per-cell resistances (can differ from crosspoint
to crosspoint), per-row input voltages, wire (line) resistance, and WL buffer
output resistance, then just run:

    python3 examples/run_custom.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from crossbar.topology import CrossbarConfig
from crossbar.compare import ideal_vmm, compare_all, print_report

# ---------------------------------------------------------------------------
# Parameters -- edit these directly.
# ---------------------------------------------------------------------------

# Crosspoint cell resistances, ohms. Nested list: one inner list per row, one
# value per column. Values may differ per cell; shape must be (n_rows, n_cols).
R_CELLS = [
    [1000.0, 1000.0, 1000.0, 1000.0],
    [1000.0,  500.0, 1000.0, 1000.0],
    [1000.0, 1000.0, 2000.0, 1000.0],
    [1000.0, 1000.0, 1000.0, 1000.0],
]

# Row input voltages, volts. One value per row -- length must equal n_rows.
V_IN = [0.7, 0.7, 0.7, 0.7]

# Wire (interconnect) resistance per segment between crosspoints, ohms.
R_ROW = 10.0  # word-line (row) segment
R_COL = 10.0  # bit-line (column) segment

# Output resistance of the WL buffers/inverters, ohms (0 = ideal buffer).
BUFFER_R_OUT = 0.0

# ---------------------------------------------------------------------------


def main() -> None:
    r = np.asarray(R_CELLS, dtype=float)
    v_in = np.asarray(V_IN, dtype=float)
    if r.ndim != 2:
        raise ValueError("R_CELLS must be a 2-D nested list: (n_rows, n_cols)")
    if v_in.shape != (r.shape[0],):
        raise ValueError(f"V_IN must have {r.shape[0]} entries (one per row), got {v_in.shape[0]}")

    g = 1.0 / r
    cfg = CrossbarConfig(g=g, v_in=v_in, r_row=R_ROW, r_col=R_COL, buffer_r_out=BUFFER_R_OUT)

    print(f"Crossbar: {cfg.n_rows}x{cfg.n_cols}, R_row={R_ROW} ohm, R_col={R_COL} ohm, "
          f"buffer_r_out={BUFFER_R_OUT} ohm")
    print(f"V_in = {v_in}")
    print(f"R_cells =\n{r}\n")

    ideal = ideal_vmm(cfg)
    results = compare_all(cfg)
    print_report(ideal, results)


if __name__ == "__main__":
    main()
