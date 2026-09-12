"""SPICE (ngspice) netlist generator for a CrossbarConfig."""
from __future__ import annotations

from .topology import CrossbarConfig


def _fmt(x: float) -> str:
    """Plain SPICE-safe numeric literal (avoids e.g. numpy's `np.float64(...)` repr)."""
    return repr(float(x))


def generate_netlist(cfg: CrossbarConfig, title: str | None = None) -> str:
    n, m = cfg.n_rows, cfg.n_cols
    lines: list[str] = [title or f"Crossbar VMM ({cfg.variant_label()}, {n}x{m})"]

    # --- Row voltage sources ---
    for i in range(n):
        lines.append(f"V{i} {cfg.row_node(i, 0)} 0 DC {_fmt(cfg.v_in[i])}")

    # --- Row (word line) wiring ---
    for i in range(n):
        for j in range(1, m):
            dst = cfg.row_node(i, j)
            src = cfg.row_node(i, j - 1)
            if cfg.is_buffered_step(j):
                # Ideal unity-gain buffer: re-drives the row to the source's
                # own node voltage (exact V_i, since the source is ideal),
                # regardless of the current drawn downstream.
                lines.append(f"Ebuf_{i}_{j} {dst} 0 {cfg.row_node(i, 0)} 0 1")
            else:
                lines.append(f"Rrow_{i}_{j} {src} {dst} {_fmt(cfg.r_row)}")

    # --- Crosspoint resistors ---
    for i in range(n):
        for j in range(m):
            r_val = 1.0 / cfg.g[i, j]
            lines.append(
                f"Rcell_{i}_{j} {cfg.row_node(i, j)} {cfg.col_node(j, i)} {_fmt(r_val)}"
            )

    # --- Column (bit line) wiring ---
    for j in range(m):
        bottom = cfg.col_bottom(j)
        if cfg.star_columns:
            # Dedicated wire per cell straight to the virtual ground node;
            # resistance scales with physical distance from the bottom.
            for i in range(n):
                dist = n - i  # segments from crosspoint i to the bottom
                lines.append(
                    f"Rcol_{j}_{i} {cfg.col_node(j, i)} {bottom} {_fmt(cfg.r_col * dist)}"
                )
        else:
            for i in range(n - 1):
                lines.append(
                    f"Rcol_{j}_{i} {cfg.col_node(j, i)} {cfg.col_node(j, i + 1)} {_fmt(cfg.r_col)}"
                )
            lines.append(
                f"Rcol_{j}_{n - 1} {cfg.col_node(j, n - 1)} {bottom} {_fmt(cfg.r_col)}"
            )
        # Virtual ground (ideal TIA): 0 V source, its current is the VMM output.
        lines.append(f"Vsense_{j} {bottom} 0 DC 0")

    lines.append(".op")
    lines.append(".control")
    lines.append("run")
    save_exprs = " ".join(f"i(Vsense_{j})" for j in range(m))
    lines.append(f"print {save_exprs}")
    lines.append(".endc")
    lines.append(".end")
    return "\n".join(lines) + "\n"
