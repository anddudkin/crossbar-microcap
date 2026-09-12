# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

An ngspice-based simulator for a resistive crossbar array used for analog
vector-matrix multiplication (VMM). It exists to compare methods of
compensating IR-drop (line voltage drop along word lines / bit lines)
against the ideal, zero-resistance VMM result — for research purposes
(PhD work / patent), so numerical correctness of the circuit model and
reproducibility of results matter more than code generality.

The project originated from Micro-Cap 12 schematics of a single resistive
word line (`reference/microcap/*.cir`), which used inverter-based repeaters
to fight line attenuation. That idea is generalized here into a full NxM
crossbar and re-implemented on ngspice (open, scriptable, Linux-friendly)
instead of Micro-Cap.

## Commands

```
sudo apt-get install ngspice        # required system dependency, not pip-installable
pip install -r requirements.txt     # numpy, matplotlib

python3 examples/run_demo.py --rows 8 --cols 8 --r-line 10 --r-cell 1000 --v-in 0.7
python3 examples/run_wl_buffer_interval_sweep.py --rows 8 --cols 8 --r-line 10 --r-cell 1000 --v-in 0.7
```

There is no test suite, linter, or build step configured. There is no
package installer config (setup.py/pyproject.toml) — the `crossbar` package
is used in-place; scripts add the repo root to `sys.path` (see
`examples/run_demo.py`) rather than relying on an installed package.

To sanity-check a change quickly, run a small case directly, e.g.:

```
python3 -c "
import numpy as np
from crossbar.topology import CrossbarConfig
from crossbar.compare import ideal_vmm, run_variant
cfg = CrossbarConfig(g=np.array([[1e-3,2e-3],[3e-3,1e-3]]), v_in=np.array([1.0,0.5]), r_row=2.0, r_col=2.0)
print(ideal_vmm(cfg), run_variant(cfg))
"
```

## Architecture

`crossbar/topology.py` (`CrossbarConfig`) is the single source of truth for
a crossbar's electrical structure — array size, conductance matrix `g`,
row input voltages `v_in`, per-segment wire resistance `r_row`/`r_col`,
and the compensation flags (`buffer_interval`, `source_buffer`,
`star_columns`). Both the SPICE netlist generator and the schematic
generator consume this same object, specifically so the simulated circuit
and the drawn figure can never drift apart — when changing the electrical
model, `topology.py` is the only place node naming/positions are decided;
`netlist.py` and `schematic.py` must stay in sync with it rather than each
inventing their own layout.

Data flow: `CrossbarConfig` → `netlist.generate_netlist()` (SPICE text) →
`simulate.run_ngspice()` (subprocess, batch mode `-b`) →
`simulate.read_column_currents()` (parses `.op` `print` output) →
`compare.py` turns that into error metrics against `compare.ideal_vmm()`
(the zero-resistance closed-form result, `v_in @ g`, no simulator needed).

Circuit model (see full description in `README.md`): rows are ideal
voltage sources with series wire resistance between crosspoints; columns
terminate in an ideal transimpedance amplifier modeled as a 0V source to
ground, whose branch current is directly the VMM output current for that
column — this is why `read_column_currents` reads `i(Vsense_j)` without
needing an op-amp subcircuit.

Compensation methods are not separate code paths but flags on the same
`CrossbarConfig`/netlist generator:
- `buffer_interval > 0` replaces a passive `Rrow` segment with an ideal
  unity-gain VCVS (`Ebuf_*`) at every Nth crosspoint, referenced back to
  the row's own source node (`rn_i_0`) — a repeater that re-drives the
  exact intended `V_i` regardless of downstream current. Kept general for
  a *periodic/partial* buffering study (interval > 1) — see
  `examples/run_wl_buffer_interval_sweep.py` — not part of the main
  `compare_all` variant set.
- `source_buffer = True` puts an ideal buffer between the row's raw
  voltage source and crosspoint 0 (a `vsrc_i` node plus `Ebufsrc_i`),
  explicit even though the source is already ideal (so it stays meaningful
  once a non-ideal source is modelled later). Combined with
  `buffer_interval = 1` (a buffer before every crosspoint) this is *full*
  WL compensation (`variant_label()` reports it as `wl_buffer_full`) — the
  current experiment's row-side method, electrically zeroing row IR-drop.
- `star_columns = True` replaces the shared chain of `Rcol` segments with
  a dedicated resistor from each crosspoint straight to the column's
  virtual ground (length/resistance scales with distance from the
  bottom), removing shared-wire coupling between cells in a column.
- `compare.compare_all` runs four combinations of the *full* WL method with
  `star_columns`: baseline / wl_buffer_full / bl_star / combined_full.
  `compare.evaluate_variant` is the reusable single-variant runner (ngspice
  run + error metrics) both `compare_all` and the interval-sweep script
  build on, so a new variant set doesn't need to duplicate that plumbing.

`crossbar/schematic.py` draws the regular grid structure directly from
`CrossbarConfig` (matplotlib, saved as SVG+PNG) rather than laying out the
generated netlist — general netlist-to-schematic auto-layout produces
unreadable results for anything but the most regular circuits, so schematic
generation deliberately mirrors the topology model's node placement instead.
It is meant for illustrative small subsets (e.g. 4x4); full-size arrays are
simulated but not schematically rendered in full. Every wire segment that
carries `R_row`/`R_col` is drawn as an explicit resistor zigzag (distinct,
smaller-amplitude style from the crosspoint cell resistors) — a buffered
row segment draws the buffer symbol instead of a zigzag, since it has zero
effective resistance. Don't let this drift back to plain lines for
interconnect — that previously made it look like wire resistance wasn't
modelled at all, which it always was (in the netlist) even when the
picture didn't show it.

`examples/run_demo.py` is the orchestration script (CLI flags for array
size, line/cell resistance, input voltage) and the reference for how the
pieces above compose end-to-end for the *full*-compensation experiment,
including writing `out/vmm_comparison.csv` and `out/error_comparison.png`.
Its default array is uniform (`g` and `v_in` constant), not random —
deliberately, so any asymmetry across rows/columns in the results comes
only from wire position, isolating the IR-drop effect being studied.
`examples/run_wl_buffer_interval_sweep.py` is the separate script for the
periodic-buffering research question (`buffer_interval` swept,
`source_buffer=False`); it's independent of `compare_all` on purpose so
that experiment doesn't get entangled with the main one. `out/` is
gitignored — it is regenerated output, not checked-in state.

## Extending the cell model

Cells are currently ideal linear resistors (`Rcell_*` in
`netlist.generate_netlist`). Swapping in a nonlinear memristor model means
replacing those lines with a SPICE behavioral (`B`) source or compiled
device model, and likely generalizing the analysis from `.op` to
`.dc`/`.tran` if the device is history-dependent (the current model is
purely resistive, so a single operating-point analysis is exact and
sufficient).
