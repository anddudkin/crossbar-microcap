"""Independent nodal-analysis solver for CrossbarConfig.

Builds and solves the exact same resistor network described by
crossbar.netlist.generate_netlist directly via linear algebra (Modified
Nodal Analysis, MNA) — no ngspice involved. This mirrors generate_netlist
element-for-element (same nodes, same buffer/star conditionals) so it is a
genuinely independent implementation of the same electrical model:
agreement with crossbar.simulate's ngspice results is a correctness check
on the SPICE netlist, not merely an approximation — the underlying network
is linear, so both methods solve the same equations exactly (up to
floating-point/solver tolerance).

Two solver backends are provided, both exact (same equations, same
answer) and kept side by side rather than one replacing the other:

- `solve_analytic` / `_MNA`: dense `numpy.linalg.solve`. Simple, fine up to
  a few thousand nodes (roughly 32x32-64x64 crossbars) — see
  examples/validate_analytic.py.
- `solve_analytic_sparse` / `_MNASparse`: the same stamps assembled into a
  `scipy.sparse` matrix and solved with `spsolve`. The network is very
  sparse (every node touches only a handful of neighbours), so this scales
  to much larger arrays (128x128+) where the dense O(n^3) solve becomes
  too slow and memory-hungry (dense solve on a 128x128 array was observed
  to exhaust memory).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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


class _MNASparse:
    """Sparse-matrix counterpart of `_MNA` — identical stamps and the same
    `resistor`/`vsource`/`vcvs`/`solve` interface, so it's a drop-in
    replacement wherever a builder object is expected (see `_populate`).
    Kept as a separate class rather than folded into `_MNA` so the simple
    dense path stays untouched.
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
        A = sp.lil_matrix((size, size))
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

        for k, (node_p, value) in enumerate(self._vsources):
            row = n + k
            if node_p is not None:
                A[node_p, row] -= 1.0
                A[row, node_p] += 1.0
            z[row] = value

        for k, (node_out, node_ctrl, gain) in enumerate(self._vcvs):
            row = n + nv + k
            if node_out is not None:
                A[node_out, row] -= 1.0
                A[row, node_out] += 1.0
            if node_ctrl is not None:
                A[row, node_ctrl] -= gain
            z[row] = 0.0

        x = spla.spsolve(A.tocsc(), z)
        return {
            tag: (x[n + idx] if kind == "v" else x[n + nv + idx])
            for tag, (kind, idx) in self._tags.items()
        }


def _add_buffer(mna, name: str, dst: str, src: str, r_out: float) -> None:
    """Mirrors netlist._buffer_lines: an ideal VCVS, optionally through an
    explicit output resistor instead of driving `dst` directly."""
    if r_out > 0:
        inode = f"{name}_out"
        mna.vcvs(inode, src, 1.0)
        mna.resistor(inode, dst, r_out)
    else:
        mna.vcvs(dst, src, 1.0)


def _add_comparator_buffer(mna, name: str, dst: str, level: float, r_out: float) -> None:
    """Mirrors netlist._comparator_buffer_lines: a fixed DC source at the
    precomputed HIGH/LOW `level`, optionally through an explicit output
    resistor, instead of a VCVS copying `src`."""
    if r_out > 0:
        inode = f"{name}_out"
        mna.vsource(inode, level)
        mna.resistor(inode, dst, r_out)
    else:
        mna.vsource(dst, level)


def _add_buffer_or_comparator(cfg: CrossbarConfig, mna, i: int, name: str, dst: str, src: str) -> None:
    """Dispatches to the ideal linear buffer or the comparator repeater,
    whichever cfg.comparator_buffer selects — mirrors
    netlist._buffer_or_comparator_lines."""
    if cfg.comparator_buffer:
        _add_comparator_buffer(mna, name, dst, cfg.buffer_output_level(i), cfg.buffer_r_out)
    else:
        _add_buffer(mna, name, dst, src, cfg.buffer_r_out)


def _populate(cfg: CrossbarConfig, mna) -> None:
    """Adds every element of `cfg`'s network to `mna` (either `_MNA` or
    `_MNASparse` — both expose the same resistor/vsource/vcvs/solve
    interface). Same element-by-element structure as
    crossbar.netlist.generate_netlist — see that module for the physical
    reasoning behind each conditional. Shared between both solver backends
    so they can never drift apart from each other."""
    n, m = cfg.n_rows, cfg.n_cols

    for i in range(n):
        if cfg.source_buffer:
            src_node = f"vsrc_{i}"
            mna.vsource(src_node, float(cfg.v_in[i]))
            _add_buffer_or_comparator(cfg, mna, i, f"bufsrc_{i}", cfg.row_node(i, 0), src_node)
        else:
            mna.vsource(cfg.row_node(i, 0), float(cfg.v_in[i]))

    for i in range(n):
        for j in range(1, m):
            dst, src = cfg.row_node(i, j), cfg.row_node(i, j - 1)
            if cfg.is_buffered_step(j):
                _add_buffer_or_comparator(cfg, mna, i, f"buf_{i}_{j}", dst, cfg.row_node(i, 0))
            else:
                mna.resistor(src, dst, cfg.r_row)

    for i in range(n):
        for j in range(m):
            mna.resistor(cfg.row_node(i, j), cfg.col_node(j, i), 1.0 / cfg.g[i, j])

    for j in range(m):
        bottom = cfg.col_bottom(j)
        if cfg.r_col_end > 0:
            end = cfg.col_end(j)
            mna.resistor(end, bottom, cfg.r_col_end)
        else:
            end = bottom

        if cfg.star_columns:
            for i in range(n):
                dist = n - i
                mna.resistor(cfg.col_node(j, i), end, cfg.r_col * dist)
        else:
            for i in range(n - 1):
                mna.resistor(cfg.col_node(j, i), cfg.col_node(j, i + 1), cfg.r_col)
            mna.resistor(cfg.col_node(j, n - 1), end, cfg.r_col)
        mna.vsource(bottom, 0.0, tag=f"vsense_{j}")


def _column_currents(mna, m: int) -> np.ndarray:
    raw = mna.solve()
    # Vsense sinks whatever the array delivers: physical column current
    # (into the sense node from the array) is the negative of "current the
    # source supplies into that node" (see _MNA.solve docstring).
    return np.array([-raw[f"vsense_{j}"] for j in range(m)])


def solve_analytic(cfg: CrossbarConfig) -> np.ndarray:
    """Column output currents for `cfg`, computed by direct dense MNA
    instead of ngspice. Fine up to a few thousand nodes; for larger arrays
    use `solve_analytic_sparse` instead."""
    mna = _MNA()
    _populate(cfg, mna)
    return _column_currents(mna, cfg.n_cols)


def solve_analytic_sparse(cfg: CrossbarConfig) -> np.ndarray:
    """Same exact computation as `solve_analytic`, but assembled as a
    scipy.sparse matrix and solved with `spsolve` — scales to much larger
    arrays (128x128+) where the dense solve is too slow/memory-hungry."""
    mna = _MNASparse()
    _populate(cfg, mna)
    return _column_currents(mna, cfg.n_cols)
