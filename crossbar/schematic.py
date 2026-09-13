"""Publication-quality schematic rendering for a CrossbarConfig.

Draws the crossbar directly from its regular NxM structure (not an
auto-layout of the SPICE netlist), so the picture stays legible and matches
exactly what `crossbar.netlist.generate_netlist` simulates. Intended for
figures in a paper / patent, not as an editable EDA schematic.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

from .topology import CrossbarConfig

_COL_PITCH = 1.8 # bumped slightly over 1.3 so the star-column protected lane band (see draw_crossbar) is a third wider
_ROW_PITCH = 1.25
_CELL_DROP = 0.75  # fraction of _ROW_PITCH from the row wire down to the column continuation
_LEFT_MARGIN = 1.7
_BOTTOM_MARGIN = 1.3
_TOP_MARGIN = 0.5
_END_R_EXTRA = 0.5  # extra bottom space to fit a visibly separate r_col_end zigzag
_BUFFER_HALF_W = 0.32 * 2 / 3  # buffer triangle half-width/height, shrunk a third from the original 0.32/0.28
_BUFFER_HALF_H = 0.28 * 2 / 3


def _bottom_y(cfg: CrossbarConfig, row_y: list[float]) -> float:
    extra = _END_R_EXTRA if cfg.r_col_end > 0 else 0.0
    return row_y[-1] + _BOTTOM_MARGIN + extra


def _zigzag(ax, x0, y0, x1, y1, n=6, amp=0.09, lw=1.1, **kw):
    """Resistor zigzag symbol between two points."""
    length = np.hypot(x1 - x0, y1 - y0)
    if length < 1e-9:
        return
    t = np.linspace(0.0, 1.0, 2 * n + 1)
    xs = x0 + t * (x1 - x0)
    ys = y0 + t * (y1 - y0)
    nx, ny = -(y1 - y0) / length, (x1 - x0) / length
    offsets = np.zeros_like(t)
    offsets[1:-1] = [amp if k % 2 else -amp for k in range(1, 2 * n)]
    xs = xs + offsets * nx
    ys = ys + offsets * ny
    ax.plot(xs, ys, color="black", lw=lw, solid_capstyle="round", **kw)


def _vsource(ax, x, y, label, source_buffer=False, buffer_r_out=0.0):
    """Circle voltage source, drawn to the left feeding into (x, y).

    If `source_buffer`, a unity-gain buffer is inserted on the wire between
    the source and (x, y) — the explicit "buffer right at the source" stage
    (see CrossbarConfig.source_buffer). `buffer_r_out` > 0 draws its output
    resistance as a zigzag instead of a plain wire out of the buffer.
    """
    cx = x - 0.85 if source_buffer else x - 0.55
    ax.add_patch(Circle((cx, y), 0.22, fill=False, lw=1.2, color="black"))
    ax.text(cx, y + 0.02, "+", ha="center", va="center", fontsize=7)
    if source_buffer:
        bx = (cx + x) / 2 + 0.1
        ax.plot([cx + 0.22, bx - _BUFFER_HALF_W], [y, y], color="black", lw=1.0)
        if buffer_r_out > 0:
            _zigzag(ax, bx + _BUFFER_HALF_W, y, x, y, n=3, amp=0.05, lw=0.8)
        else:
            ax.plot([bx + _BUFFER_HALF_W, x], [y, y], color="black", lw=1.0)
        _buffer(ax, bx, y)
    else:
        ax.plot([cx + 0.22, x], [y, y], color="black", lw=1.0)
    ax.text(cx - 0.3, y, label, ha="right", va="center", fontsize=8)


def _buffer(ax, x, y):
    """Unity-gain buffer (op-amp triangle) symbol centred at (x, y)."""
    w, h = _BUFFER_HALF_W, _BUFFER_HALF_H
    tri = Polygon(
        [(x - w, y - h), (x - w, y + h), (x + w, y)],
        closed=True,
        fill=True,
        facecolor="white",
        edgecolor="black",
        lw=1.1,
    )
    ax.add_patch(tri)
    ax.text(x, y, "1x", ha="center", va="center", fontsize=6)


def _memristor_box(ax, x, y0, y1, label):
    """Memristor cell symbol: a labelled box with a small pulse-waveform
    icon, matching the crossbar-array figures common in the ReRAM/memristor
    literature (a box on the crosspoint rather than a plain resistor
    zigzag) instead of a generic resistor symbol.

    The top lead is deliberately long enough to clear the word-line
    resistor zigzag at y0 before the label starts, so the label never
    overlaps it.
    """
    h, w = 0.3, 0.5
    top_stub = 0.42
    box_top = y0 + top_stub
    box_bottom = box_top + h
    ymid = box_top + h / 2
    ax.plot([x, x], [y0, box_top], color="black", lw=1.0)
    ax.plot([x, x], [box_bottom, y1], color="black", lw=1.0)
    ax.add_patch(
        Rectangle((x - w / 2, box_top), w, h, fill=True, facecolor="white", edgecolor="black", lw=1.1)
    )
    # tiny rectangular-pulse icon inside the box
    px = np.array([-0.16, -0.16, -0.03, -0.03, 0.09, 0.09, 0.16])
    py = np.array([0.0, 0.07, 0.07, -0.07, -0.07, 0.0, 0.0])
    ax.plot(x + px, ymid + py, color="black", lw=0.8)
    ax.text(x, box_top - 0.06, label, ha="center", va="bottom", fontsize=6.5)


def _tia(ax, x, y_top, y_bottom, label):
    """Virtual-ground column termination: the column wire (last resistor already
    drawn above y_top) runs straight down to a ground symbol, no amplifier
    triangle or current-arrow — just the wire ending in ground."""
    ax.plot([x, x], [y_top, y_bottom], color="black", lw=1.0)
    for k, wid in enumerate((0.16, 0.1, 0.05)):
        gy = y_bottom + 0.1 + k * 0.09
        ax.plot([x - wid, x + wid], [gy, gy], color="black", lw=1.0)
    ax.text(x + 0.25, y_bottom + 0.1, label, ha="left", va="center", fontsize=7)


def draw_crossbar(cfg: CrossbarConfig, ax=None, title: str | None = None):
    n, m = cfg.n_rows, cfg.n_cols
    if ax is None:
        row_y_tmp = [_TOP_MARGIN + i * _ROW_PITCH for i in range(n)]
        _, ax = plt.subplots(
            figsize=(_LEFT_MARGIN + m * _COL_PITCH + 1.5, _bottom_y(cfg, row_y_tmp) + 0.5)
        )

    col_x = [_LEFT_MARGIN + j * _COL_PITCH for j in range(m)]
    row_y = [_TOP_MARGIN + i * _ROW_PITCH for i in range(n)]  # row 0 at top after invert
    bottom_y = _bottom_y(cfg, row_y)

    # --- rows (word lines) ---
    for i in range(n):
        y = row_y[i]
        _vsource(
            ax, col_x[0], y, f"V{i}={cfg.v_in[i]:g}V",
            source_buffer=cfg.source_buffer, buffer_r_out=cfg.buffer_r_out,
        )
        for j in range(1, m):
            x0, x1 = col_x[j - 1], col_x[j]
            if cfg.is_buffered_step(j):
                bx = (x0 + x1) / 2
                ax.plot([x0, bx - _BUFFER_HALF_W], [y, y], color="black", lw=1.0)
                if cfg.buffer_r_out > 0:
                    _zigzag(ax, bx + _BUFFER_HALF_W, y, x1, y, n=3, amp=0.05, lw=0.8)
                else:
                    ax.plot([bx + _BUFFER_HALF_W, x1], [y, y], color="black", lw=1.0)
                _buffer(ax, bx, y)
            else:
                # Interconnect (word-line) wire resistance R_row.
                _zigzag(ax, x0, y, x1, y, n=3, amp=0.055, lw=0.9)
        ax.plot([col_x[-1], col_x[-1] + 0.35], [y, y], color="black", lw=1.0)

    # --- crosspoint memristor cells ---
    for i in range(n):
        for j in range(m):
            x, y = col_x[j], row_y[i]
            _memristor_box(ax, x, y, y + _ROW_PITCH * _CELL_DROP, f"G{i + 1},{j + 1}")

    # --- columns (bit lines) ---
    has_end_r = cfg.r_col_end > 0
    last_cell_bottom_y = row_y[-1] + _ROW_PITCH * _CELL_DROP
    # Shared merge/end-node height: midpoint of the space below the last row,
    # so the per-cell zigzag and the r_col_end zigzag each get roughly equal,
    # visually distinguishable room (rather than a fixed offset from
    # bottom_y, which would leave the r_col_end zigzag cramped).
    end_y = (last_cell_bottom_y + bottom_y) / 2 if has_end_r else bottom_y - 0.35
    for j in range(m):
        x = col_x[j]
        cell_bottom_y = [row_y[i] + _ROW_PITCH * _CELL_DROP for i in range(n)]
        if cfg.star_columns:
            # Individual dedicated wire per cell: jog right into its own lane,
            # run down, then jog back to a common bus just above the ground
            # symbol so the picture makes clear these never share a segment.
            # The lane band is capped so it never reaches whatever sits to
            # the right of this column (the next column's own crosspoint
            # box, or — if this row segment is buffered — the buffer
            # triangle on the way to it), so a fan-out wire never crosses an
            # element that isn't its own; within that band, the n lanes are
            # spaced as tight as the band allows.
            box_half = 0.25  # _memristor_box half-width
            # A straight line needs no lateral clearance for wiggle (unlike a
            # zigzag), so the lane margin can be tighter than the crossing
            # check alone would require.
            lane_margin = 0.015
            min_off = box_half + lane_margin
            if j < m - 1 and cfg.is_buffered_step(j + 1):
                max_off = _COL_PITCH / 2 - _BUFFER_HALF_W - lane_margin
            else:
                max_off = _COL_PITCH - box_half - lane_margin
            max_off = max(max_off, min_off + 0.01)
            step = (max_off - min_off) / (n - 1) if n > 1 else 0.0
            merge_y = end_y
            for i in range(n):
                fan_x = x + min_off + step * i
                ax.plot([x, fan_x], [cell_bottom_y[i], cell_bottom_y[i]], color="black", lw=0.9)
                # Dedicated per-cell wire resistance R_col * distance-to-bottom,
                # drawn as a plain straight lead (not the usual zigzag) so the
                # tightly packed star lanes stay visually parallel and distinct.
                ax.plot([fan_x, fan_x], [cell_bottom_y[i], merge_y], color="black", lw=0.8)
                ax.plot([fan_x, x], [merge_y, merge_y], color="black", lw=0.9)
            top_for_tia = merge_y
        else:
            for i in range(n - 1):
                # Interconnect (bit-line) wire resistance R_col.
                _zigzag(ax, x, cell_bottom_y[i], x, row_y[i + 1], n=3, amp=0.055, lw=0.9)
            last_y = end_y if has_end_r else bottom_y
            _zigzag(ax, x, cell_bottom_y[-1], x, last_y, n=3, amp=0.055, lw=0.9)
            top_for_tia = last_y
        if has_end_r:
            # Shared bit-line-end resistance (r_col_end), common to every
            # cell in this column, distinct from the per-cell R_col zigzags.
            # A short plain lead separates the two zigzags visually so they
            # don't read as one longer resistor.
            gap = 0.08
            ax.plot([x, x], [top_for_tia, top_for_tia + gap], color="black", lw=1.0)
            _zigzag(ax, x, top_for_tia + gap, x, bottom_y, n=3, amp=0.06, lw=1.0)
            ax.text(x + 0.12, (top_for_tia + gap + bottom_y) / 2, "R_end", ha="left", va="center", fontsize=6)
            top_for_tia = bottom_y
        _tia(ax, x, top_for_tia, bottom_y, f"I{j}")

    ax.set_xlim(0, col_x[-1] + 1.5)
    ax.set_ylim(bottom_y + 0.5, -0.4)  # inverted: row 0 at top
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        title or f"{cfg.variant_label()} ({n}x{m}), R_line={cfg.r_row:g}Ω",
        fontsize=10,
    )
    return ax


def save_schematic(cfg: CrossbarConfig, path_svg: str, path_png: str | None = None, title: str | None = None) -> None:
    row_y = [_TOP_MARGIN + i * _ROW_PITCH for i in range(cfg.n_rows)]
    fig_w = _LEFT_MARGIN + cfg.n_cols * _COL_PITCH + 1.5
    fig_h = _bottom_y(cfg, row_y) + 0.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    draw_crossbar(cfg, ax=ax, title=title)
    fig.tight_layout()
    fig.savefig(path_svg)
    if path_png:
        fig.savefig(path_png, dpi=200)
    plt.close(fig)
