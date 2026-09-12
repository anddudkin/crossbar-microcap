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
from matplotlib.patches import Circle, FancyArrow, Polygon

from .topology import CrossbarConfig

_COL_PITCH = 1.6
_ROW_PITCH = 1.2
_LEFT_MARGIN = 2.0
_BOTTOM_MARGIN = 1.6
_TOP_MARGIN = 0.6


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


def _vsource(ax, x, y, label, source_buffer=False):
    """Circle voltage source, drawn to the left feeding into (x, y).

    If `source_buffer`, an ideal unity-gain buffer is inserted on the wire
    between the source and (x, y) — the explicit "buffer right at the
    source" stage (see CrossbarConfig.source_buffer).
    """
    cx = x - 0.85 if source_buffer else x - 0.55
    ax.add_patch(Circle((cx, y), 0.22, fill=False, lw=1.2, color="black"))
    ax.text(cx, y + 0.02, "+", ha="center", va="center", fontsize=7)
    if source_buffer:
        bx = (cx + x) / 2 + 0.1
        ax.plot([cx + 0.22, bx - 0.32], [y, y], color="black", lw=1.0)
        ax.plot([bx + 0.32, x], [y, y], color="black", lw=1.0)
        _buffer(ax, bx, y)
    else:
        ax.plot([cx + 0.22, x], [y, y], color="black", lw=1.0)
    ax.text(cx - 0.3, y, label, ha="right", va="center", fontsize=8)


def _buffer(ax, x, y):
    """Unity-gain buffer (op-amp triangle) symbol centred at (x, y)."""
    w, h = 0.32, 0.28
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


def _tia(ax, x, y_top, y_bottom, label):
    """Virtual-ground / transimpedance-amplifier symbol at the column bottom."""
    ax.plot([x, x], [y_top, y_bottom + 0.15], color="black", lw=1.0)
    w, h = 0.3, 0.26
    tri = Polygon(
        [(x - w, y_bottom + 0.15 + h), (x - w, y_bottom + 0.15 - h), (x + w, y_bottom + 0.15)],
        closed=True,
        fill=True,
        facecolor="white",
        edgecolor="black",
        lw=1.1,
    )
    ax.add_patch(tri)
    gx, gy = x + w, y_bottom + 0.15
    for k, wid in enumerate((0.18, 0.11, 0.05)):
        ax.plot([gx + 0.12 + k * 0.09] * 2, [gy - wid, gy + wid], color="black", lw=1.0)
    arrow_y = y_bottom
    ax.add_patch(
        FancyArrow(
            x, arrow_y - 0.02, 0, -0.35, width=0.01, head_width=0.09, head_length=0.12,
            color="black", length_includes_head=True,
        )
    )
    ax.text(x + 0.05, arrow_y - 0.55, label, ha="center", va="top", fontsize=7)


def draw_crossbar(cfg: CrossbarConfig, ax=None, title: str | None = None):
    n, m = cfg.n_rows, cfg.n_cols
    if ax is None:
        _, ax = plt.subplots(
            figsize=(_LEFT_MARGIN + m * _COL_PITCH + 1.5, _TOP_MARGIN + n * _ROW_PITCH + _BOTTOM_MARGIN)
        )

    col_x = [_LEFT_MARGIN + j * _COL_PITCH for j in range(m)]
    row_y = [_TOP_MARGIN + i * _ROW_PITCH for i in range(n)]  # row 0 at top after invert
    bottom_y = row_y[-1] + _BOTTOM_MARGIN

    # --- rows (word lines) ---
    for i in range(n):
        y = row_y[i]
        _vsource(ax, col_x[0], y, f"V{i}={cfg.v_in[i]:g}V", source_buffer=cfg.source_buffer)
        for j in range(1, m):
            x0, x1 = col_x[j - 1], col_x[j]
            if cfg.is_buffered_step(j):
                bx = (x0 + x1) / 2
                ax.plot([x0, bx - 0.32], [y, y], color="black", lw=1.0)
                ax.plot([bx + 0.32, x1], [y, y], color="black", lw=1.0)
                _buffer(ax, bx, y)
            else:
                # Interconnect (word-line) wire resistance R_row.
                _zigzag(ax, x0, y, x1, y, n=4, amp=0.06, lw=0.9)
        ax.plot([col_x[-1], col_x[-1] + 0.35], [y, y], color="black", lw=1.0)

    # --- crosspoint cell resistors ---
    for i in range(n):
        for j in range(m):
            x, y = col_x[j], row_y[i]
            _zigzag(ax, x, y, x, y + _ROW_PITCH * 0.55)

    # --- columns (bit lines) ---
    for j in range(m):
        x = col_x[j]
        cell_bottom_y = [row_y[i] + _ROW_PITCH * 0.55 for i in range(n)]
        if cfg.star_columns:
            # Individual dedicated wire per cell: jog right into its own lane,
            # run down, then jog back to a common bus just above the TIA so
            # the picture makes clear these never share a segment.
            merge_y = bottom_y - 0.35
            for i in range(n):
                fan_x = x + 0.16 + 0.11 * i
                ax.plot([x, fan_x], [cell_bottom_y[i], cell_bottom_y[i]], color="black", lw=0.9)
                # Dedicated per-cell wire resistance R_col * distance-to-bottom.
                _zigzag(ax, fan_x, cell_bottom_y[i], fan_x, merge_y, n=3, amp=0.05, lw=0.8)
                ax.plot([fan_x, x], [merge_y, merge_y], color="black", lw=0.9)
            top_for_tia = merge_y
        else:
            for i in range(n - 1):
                # Interconnect (bit-line) wire resistance R_col.
                _zigzag(ax, x, cell_bottom_y[i], x, row_y[i + 1], n=4, amp=0.06, lw=0.9)
            _zigzag(ax, x, cell_bottom_y[-1], x, bottom_y, n=4, amp=0.06, lw=0.9)
            top_for_tia = cell_bottom_y[-1]
        _tia(ax, x, top_for_tia, bottom_y, f"I{j}")

    ax.set_xlim(0, col_x[-1] + 1.5)
    ax.set_ylim(bottom_y + 0.9, -0.4)  # inverted: row 0 at top
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        title or f"{cfg.variant_label()} ({n}x{m}), R_line={cfg.r_row:g}Ω",
        fontsize=10,
    )
    return ax


def save_schematic(cfg: CrossbarConfig, path_svg: str, path_png: str | None = None, title: str | None = None) -> None:
    fig_w = _LEFT_MARGIN + cfg.n_cols * _COL_PITCH + 1.5
    fig_h = _TOP_MARGIN + cfg.n_rows * _ROW_PITCH + _BOTTOM_MARGIN
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    draw_crossbar(cfg, ax=ax, title=title)
    fig.tight_layout()
    fig.savefig(path_svg)
    if path_png:
        fig.savefig(path_png, dpi=200)
    plt.close(fig)
