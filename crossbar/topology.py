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
  row's IR-drop is fully eliminated — assuming an ideal (zero output
  impedance) buffer. `buffer_r_out` makes this non-ideal: every buffer
  (source and per-crosspoint) gets that series output resistance instead
  of driving its node directly, so a real op-amp/inverter's finite drive
  strength can be modelled instead of an idealized one.
- star_columns = True: each crosspoint gets its own dedicated wire straight
  to the column's virtual ground (length proportional to its distance from
  the bottom), instead of a shared chain of R_col segments. This removes
  cross-cell coupling through shared bit-line resistance.
- r_col_end (default 0): resistance of the final bit-line stretch that
  every cell in a column shares on the way to the sense amplifier — the
  routing from the edge of the array to the peripheral read circuit, which
  is physically a separate piece of wire from the per-cell R_col pitch
  (and, in star_columns mode, is the one segment every cell's otherwise
  independent dedicated wire still has in common). 0 omits it entirely
  (no extra node/resistor emitted, so existing netlists are unaffected).
  Note this reintroduces some cross-cell coupling in star_columns mode
  (all cells in a column share this one resistor), so combined_full's
  closed-form formula (see examples/validate_analytic.py) needs the
  generalized version once r_col_end > 0.
- comparator_buffer = True: every buffer that source_buffer/buffer_interval
  would otherwise place (see above) is instead a threshold comparator/
  inverter repeater — the electrical model used in the original
  reference/microcap/base_r_line_10_WL_invertor.cir (`.SUBCKT INV`:
  `E1 3 0 VALUE = { IF ( V(1)>0.35V, 100m, 0.7) }` plus a series output
  resistor), for a *fixed-amplitude* (pulsed/spiking) signalling scheme
  rather than continuous analog VMM: instead of linearly reproducing
  whatever V_i is, it snaps the row to one of two fixed rails
  (buffer_v_low/buffer_v_high) depending on which side of
  buffer_threshold V_i falls on. Since every buffer here is already
  referenced to the row's own *ideal, undropped* source node (V_i is a
  known Python float at netlist-build time, not a simulator unknown), this
  HIGH/LOW decision is precomputed once (see buffer_output_level below)
  and stamped as an ordinary fixed DC source — mathematically identical to
  evaluating the real IF(...) expression at that same ideal node, but
  strictly linear, so it stays exactly solvable by the dense/sparse
  analytic backends too (they cannot stamp a genuine nonlinear
  comparator). buffer_r_out still applies, mirroring the INV subckt's
  series output resistor. This is a separate, self-contained experiment
  (see examples/run_pulsed_buffer_demo.py) — not part of compare_all's
  four analog-VMM variants.

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
    buffer_r_out: float = 0.0  # ohms of output (series) resistance for every buffer; 0 = ideal
    star_columns: bool = False  # True enables per-cell dedicated column wiring
    r_col_end: float = 0.0  # ohms of shared bit-line-end resistance before the sense amp; 0 = none
    comparator_buffer: bool = False  # True: buffers are a threshold comparator/inverter repeater (fixed-amplitude/pulsed signalling), not a linear unity-gain VCVS
    buffer_threshold: float = 0.35  # volts; comparator decision point (default matches reference/microcap's .SUBCKT INV)
    buffer_v_low: float = 0.1  # volts; comparator output rail when V_i <= buffer_threshold
    buffer_v_high: float = 0.7  # volts; comparator output rail when V_i > buffer_threshold
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
        if self.buffer_r_out < 0:
            raise ValueError("buffer_r_out must be >= 0")
        if self.r_col_end < 0:
            raise ValueError("r_col_end must be >= 0")
        if self.buffer_v_low >= self.buffer_v_high:
            raise ValueError("buffer_v_low must be < buffer_v_high")

    @property
    def n_rows(self) -> int:
        return self.g.shape[0]

    @property
    def n_cols(self) -> int:
        return self.g.shape[1]

    def variant_label(self) -> str:
        wl = self.source_buffer and self.buffer_interval == 1
        if wl and self.star_columns:
            label = "combined_full"
        elif wl:
            label = "wl_buffer_full"
        elif self.buffer_interval > 0 and self.star_columns:
            label = "combined_periodic"
        elif self.buffer_interval > 0:
            label = "wl_buffer_periodic"
        elif self.star_columns:
            label = "bl_star"
        else:
            label = "baseline"
        return f"{label}_pulsed" if self.comparator_buffer else label

    def buffer_output_level(self, i: int) -> float:
        """The comparator/inverter repeater's precomputed HIGH/LOW output for
        row i (see comparator_buffer above). Every buffer in this project is
        referenced to the row's own ideal, undropped source node, so V_i is
        already a known value at netlist-build time — this is the single
        place that decision is made, shared by netlist.py and analytic.py so
        they can't drift apart."""
        return self.buffer_v_high if self.v_in[i] > self.buffer_threshold else self.buffer_v_low

    def row_node(self, i: int, j: int) -> str:
        """SPICE node name for row i at crosspoint j (0-indexed)."""
        return f"rn_{i}_{j}"

    def col_node(self, j: int, i: int) -> str:
        """SPICE node name for column j at crosspoint i (0-indexed)."""
        return f"cn_{j}_{i}"

    def col_bottom(self, j: int) -> str:
        return f"bot_{j}"

    def col_end(self, j: int) -> str:
        """Node where every cell in column j's wiring meets the shared
        bit-line-end resistor (r_col_end), before the sense amp. Only
        meaningful (i.e. distinct from col_bottom) when r_col_end > 0."""
        return f"blend_{j}"

    def is_buffered_step(self, j: int) -> bool:
        """Whether the transition into crosspoint j (from j-1) uses a buffer."""
        return self.buffer_interval > 0 and j > 0 and j % self.buffer_interval == 0
