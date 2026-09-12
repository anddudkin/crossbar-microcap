"""Parametric model of a resistive crossbar array for VMM (vector-matrix multiplication).

This is the single source of truth for a crossbar's electrical structure: it is
consumed both by the SPICE netlist generator (crossbar.netlist) and by the
schematic generator (crossbar.schematic), so the simulated circuit and the
drawn schematic never drift apart.

Physical model
--------------
- Rows are word lines (WL), driven by ideal voltage sources V_i at node
  rn(i, 0). Wire resistance R_row separates consecutive crosspoints along
  a row.
- Columns are bit lines (BL), terminated at the bottom by an ideal
  transimpedance amplifier modelled as a 0 V source to ground (virtual
  ground), whose current is the VMM output for that column.
- Each crosspoint (i, j) holds an ideal linear resistor of value 1/G[i, j].

Compensation methods
---------------------
- buffer_interval > 0: every `buffer_interval`-th crosspoint along a row is
  fed by an ideal unity-gain buffer (VCVS) referenced to the row's own
  source node instead of through a passive R_row segment. This re-drives
  the row to the exact intended V_i at that point, resetting accumulated
  IR-drop periodically (a repeater, as in
  reference/microcap/base_r_line_10_WL_invertor.cir). Kept for later study
  of *partial* periodic buffering (interval > 1); see
  examples/run_wl_buffer_interval_sweep.py. The *current* experiment uses
  full buffering instead (see below).
- source_buffer + buffer_interval=1: full per-cell WL compensation. A
  buffer is placed right at the row's signal source (before the first
  crosspoint) and another before every subsequent crosspoint
  (buffer_interval=1 makes every row segment a buffer instead of a passive
  R_row), so every crosspoint sees the row's intended V_i exactly and the
  row's IR-drop is fully eliminated.
- star_columns = True: each crosspoint gets its own dedicated wire straight
  to the column's virtual ground (length proportional to its distance from
  the bottom), instead of a shared chain of R_col segments. This removes
  cross-cell coupling through shared bit-line resistance.

Any of these can be combined with star_columns = True.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class CrossbarConfig:
    g: np.ndarray  # (n_rows, n_cols) conductances, Siemens
    v_in: np.ndarray  # (n_rows,) row drive voltages, Volts
    r_row: float = 10.0  # ohms per row (word line) segment
    r_col: float = 10.0  # ohms per column (bit line) segment
    buffer_interval: int = 0  # 0 disables WL buffering; else insert every N-th crosspoint
    source_buffer: bool = False  # True inserts an ideal buffer right at the row's signal source
    star_columns: bool = False  # True enables per-cell dedicated column wiring
    name: str = "crossbar"

    def __post_init__(self) -> None:
        self.g = np.asarray(self.g, dtype=float)
        self.v_in = np.asarray(self.v_in, dtype=float)
        if self.g.ndim != 2:
            raise ValueError("g must be a 2-D (n_rows, n_cols) array")
        if self.v_in.shape != (self.n_rows,):
            raise ValueError("v_in must have shape (n_rows,)")
        if np.any(self.g <= 0):
            raise ValueError("all conductances must be strictly positive")
        if self.buffer_interval < 0:
            raise ValueError("buffer_interval must be >= 0")

    @property
    def n_rows(self) -> int:
        return self.g.shape[0]

    @property
    def n_cols(self) -> int:
        return self.g.shape[1]

    def variant_label(self) -> str:
        wl = self.source_buffer and self.buffer_interval == 1
        if wl and self.star_columns:
            return "combined_full"
        if wl:
            return "wl_buffer_full"
        if self.buffer_interval > 0 and self.star_columns:
            return "combined_periodic"
        if self.buffer_interval > 0:
            return "wl_buffer_periodic"
        if self.star_columns:
            return "bl_star"
        return "baseline"

    def row_node(self, i: int, j: int) -> str:
        """SPICE node name for row i at crosspoint j (0-indexed)."""
        return f"rn_{i}_{j}"

    def col_node(self, j: int, i: int) -> str:
        """SPICE node name for column j at crosspoint i (0-indexed)."""
        return f"cn_{j}_{i}"

    def col_bottom(self, j: int) -> str:
        return f"bot_{j}"

    def is_buffered_step(self, j: int) -> bool:
        """Whether the transition into crosspoint j (from j-1) uses a buffer."""
        return self.buffer_interval > 0 and j > 0 and j % self.buffer_interval == 0
