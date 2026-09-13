"""Independent nodal-analysis solver for CrossbarConfig.

Builds and solves the exact same resistor network described by
crossbar.netlist.generate_netlist directly via linear algebra (Modified
Nodal Analysis, MNA) with numpy — no ngspice involved. This mirrors
generate_netlist element-for-element (same nodes, same buffer/star
conditionals) so it is a genuinely independent implementation of the same
electrical model: agreement with crossbar.simulate's ngspice results is a
correctness check on the SPICE netlist, not merely an approximation — the
underlying network is linear, so both methods solve the same equations
exactly (up to floating-point/solver tolerance).
"""
from __future__ import annotations

import numpy as np

from .topology import CrossbarConfig


class _MNA:
    """Minimal MNA builder for resistors, independent voltage sources, and
    VCVS elements, all referenced to a single ground node "0" — exactly
    the structure of every netlist crossbar.netlist generates.

    `solve()` returns the raw per-source "current supplied into its
    positive node" for every tagged element; sign interpretation (e.g.
    flipping it for a sink like Vsense) is left to the caller, since it
    depends on how that particular source is used in the network.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, int] = {}
        self._resistors: list[tuple[int | None, int | None, float]] = []
        self._vsources: list[tuple[int | None, float]] = []
        self._vcvs: list[tuple[int | None, int | None, float]] = []
        self._tags: dict[str, tuple[str, int]] = {}

    def _n(self, name: str) -> int | None:
        if name == "0":
            return None
        return self._nodes.setdefault(name, len(self._nodes))

    def resistor(self, a: str, b: str, r: float) -> None:
        self._resistors.append((self._n(a), self._n(b), r))

    def vsource(self, node_p: str, value: float, tag: str | None = None) -> None:
        idx = len(self._vsources)
        self._vsources.append((self._n(node_p), value))
        if tag is not None:
            self._tags[tag] = ("v", idx)

    def vcvs(self, node_out: str, node_ctrl: str, gain: float, tag: str | None = None) -> None:
        idx = len(self._vcvs)
        self._vcvs.append((self._n(node_out), self._n(node_ctrl), gain))
        if tag is not None:
            self._tags[tag] = ("e", idx)

    def solve(self) -> dict[str, float]:
        n, nv, ne = len(self._nodes), len(self._vsources), len(self._vcvs)
        size = n + nv + ne
        A = np.zeros((size, size))
        z = np.zeros(size)

        for a, b, r in self._resistors:
            g = 1.0 / r
            if a is not None:
                A[a, a] += g
            if b is not None:
                A[b, b] += g
            if a is not None and b is not None:
                A[a, b] -= g
                A[b, a] -= g

        # Independent source stamp: KCL row for node_p gets "-1 * I_k" (I_k
        # is the current the source supplies into node_p); the source's own
        # row enforces V(node_p) - V(0) = value.
        for k, (node_p, value) in enumerate(self._vsources):
            row = n + k
            if node_p is not None:
                A[node_p, row] -= 1.0
                A[row, node_p] += 1.0
            z[row] = value

        # VCVS stamp: same KCL pattern at node_out for its current I_k, and
        # its own row enforces V(node_out) - gain*V(node_ctrl) = 0.
        for k, (node_out, node_ctrl, gain) in enumerate(self._vcvs):
            row = n + nv + k
            if node_out is not None:
                A[node_out, row] -= 1.0
                A[row, node_out] += 1.0
            if node_ctrl is not None:
                A[row, node_ctrl] -= gain
            z[row] = 0.0

        x = np.linalg.solve(A, z)
        return {
            tag: (x[n + idx] if kind == "v" else x[n + nv + idx])
            for tag, (kind, idx) in self._tags.items()
        }


def _add_buffer(mna: _MNA, name: str, dst: str, src: str, r_out: float) -> None:
    """Mirrors netlist._buffer_lines: an ideal VCVS, optionally through an
    explicit output resistor instead of driving `dst` directly."""
    if r_out > 0:
        inode = f"{name}_out"
        mna.vcvs(inode, src, 1.0)
        mna.resistor(inode, dst, r_out)
    else:
        mna.vcvs(dst, src, 1.0)


def solve_analytic(cfg: CrossbarConfig) -> np.ndarray:
    """Column output currents for `cfg`, computed by direct MNA instead of
    ngspice. Same element-by-element structure as
    crossbar.netlist.generate_netlist — see that module for the physical
    reasoning behind each conditional."""
    n, m = cfg.n_rows, cfg.n_cols
    mna = _MNA()

    for i in range(n):
        if cfg.source_buffer:
            src_node = f"vsrc_{i}"
            mna.vsource(src_node, float(cfg.v_in[i]))
            _add_buffer(mna, f"Ebufsrc_{i}", cfg.row_node(i, 0), src_node, cfg.buffer_r_out)
        else:
            mna.vsource(cfg.row_node(i, 0), float(cfg.v_in[i]))

    for i in range(n):
        for j in range(1, m):
            dst, src = cfg.row_node(i, j), cfg.row_node(i, j - 1)
            if cfg.is_buffered_step(j):
                _add_buffer(mna, f"Ebuf_{i}_{j}", dst, cfg.row_node(i, 0), cfg.buffer_r_out)
            else:
                mna.resistor(src, dst, cfg.r_row)

    for i in range(n):
        for j in range(m):
            mna.resistor(cfg.row_node(i, j), cfg.col_node(j, i), 1.0 / cfg.g[i, j])

    for j in range(m):
        bottom = cfg.col_bottom(j)
        if cfg.star_columns:
            for i in range(n):
                dist = n - i
                mna.resistor(cfg.col_node(j, i), bottom, cfg.r_col * dist)
        else:
            for i in range(n - 1):
                mna.resistor(cfg.col_node(j, i), cfg.col_node(j, i + 1), cfg.r_col)
            mna.resistor(cfg.col_node(j, n - 1), bottom, cfg.r_col)
        mna.vsource(bottom, 0.0, tag=f"vsense_{j}")

    raw = mna.solve()
    # Vsense sinks whatever the array delivers: physical column current
    # (into the sense node from the array) is the negative of "current the
    # source supplies into that node" (see _MNA.solve docstring).
    return np.array([-raw[f"vsense_{j}"] for j in range(m)])
