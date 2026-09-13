# crossbar-microcap

Circuit-level modelling of a resistive crossbar array for analog
vector-matrix multiplication (VMM), and comparison of IR-drop (line
voltage drop) compensation techniques, simulated in **ngspice**.

## Background

The project started from Micro-Cap 12 schematics of a single resistive
word line (`reference/microcap/*.cir`) used to study how a repeater
(inverter buffer) chain affects line delay/attenuation. The same idea is
generalized here to a full NxM resistive crossbar, implemented with an
open, scriptable simulator (ngspice) instead of Micro-Cap, so arrays of
any size and any compensation scheme can be generated and simulated
programmatically.

## Physical model

- **Cells**: ideal linear resistors `R_ij = 1/G_ij` at each crosspoint
  (rows x columns).
- **Rows (word lines)**: driven by ideal voltage sources `V_i`; wire
  resistance `R_row` separates consecutive crosspoints along a row.
- **Columns (bit lines)**: terminated by an ideal transimpedance amplifier,
  modelled as a 0 V source to ground (virtual ground). Its branch current
  is the VMM output for that column: `I_j = sum_i V_i * G_ij` in the ideal
  (zero wire resistance) case.
- With non-zero `R_row`/`R_col`, the array becomes a resistive network
  whose exact solution is what ngspice's `.op` analysis computes — this
  is where IR-drop shows up as deviation from the ideal VMM result.

All of this lives in one place, `crossbar/topology.py` (`CrossbarConfig`),
which both the netlist generator (`crossbar/netlist.py`) and the schematic
generator (`crossbar/schematic.py`) consume, so the simulated circuit and
the drawn figure never drift apart.

## Compensation methods implemented

- **`wl_buffer_full`** — full row (word-line) compensation: a unity-gain
  buffer sits right at the row's signal source (`source_buffer=True`) and
  another before *every* crosspoint (`buffer_interval=1`), instead of a
  passive `R_row` segment. With an ideal buffer (`buffer_r_out=0`, the
  default) every crosspoint then sees the row's intended `V_i` exactly,
  fully eliminating row IR-drop. `buffer_r_out > 0` gives every buffer a
  finite output (series) resistance instead, to see how much of that
  compensation survives with a real (non-ideal) driver.
- **`bl_star`** — each crosspoint gets its own dedicated wire straight to
  the column's virtual ground (resistance proportional to its distance
  from the bottom), instead of a shared chain of `R_col` segments. This
  removes cross-cell coupling through shared bit-line resistance.
- **`combined_full`** — both of the above together.
- **`baseline`** — plain resistive crossbar, no compensation, for reference.

*Partial/periodic* WL buffering (a repeater only every `buffer_interval`-th
crosspoint, `source_buffer=False`, the original idea from
`base_r_line_10_WL_invertor.cir`) is kept as a separate, later research
question — see `examples/run_wl_buffer_interval_sweep.py` — and is not part
of the four variants above.

## Requirements

```
sudo apt-get install ngspice
pip install -r requirements.txt
```

## Usage

```
python3 examples/run_demo.py --rows 8 --cols 8 --r-line 10 --r-cell 1000 --v-in 0.7 --buffer-r-out 0
```

The default experiment uses uniform inputs and cell values (`V_in=0.7V`,
`R_cell=1kΩ`, `R_line=10Ω` per WL/BL wire segment) so any variation across
rows/columns comes purely from wire position, not from data randomness —
this isolates the IR-drop effect the compensation methods are meant to fix.
It will:
1. Build the `rows x cols` array (uniform `G`/`V_in`, or pass your own via
   the library API below).
2. Run all four variants through ngspice and compare each against the
   ideal (zero-resistance) VMM result — RMSE, max absolute and max
   relative column error (`out/vmm_comparison.csv`, printed to stdout).
3. Save a bar-chart comparison (`out/error_comparison.png`).
4. Render illustrative schematics of a small `NxN` subset for all four
   variants (`out/schematics/*.svg` and `.png`) — vector figures suitable
   for a paper or patent, drawn directly from the array topology rather
   than auto-laid-out from the netlist. Wire resistance (`R_row`/`R_col`)
   is drawn as an explicit zigzag on every segment, distinct from the
   crosspoint cell resistors, so it's visible that it's actually modelled.

For the periodic-buffering sweep (kept separate, see above):

```
python3 examples/run_wl_buffer_interval_sweep.py --rows 8 --cols 8 --r-line 10 --r-cell 1000 --v-in 0.7 --intervals 1 2 3 4 8
```

## Validating the SPICE model against an analytical solution

`crossbar/analytic.py` solves the *exact same* resistor network directly
via Modified Nodal Analysis (numpy linear algebra), independently of
ngspice — same nodes, same buffer/star conditionals as
`crossbar/netlist.py`, just solved a different way. Since the network is
linear, this isn't an approximation: ngspice and the MNA solver should
agree to numerical precision, and disagreement would mean a bug in one of
the two independent implementations. For `combined_full` specifically,
there's also a closed-form expression (row IR-drop is fully cancelled by
WL buffering, and star columns remove cross-cell coupling, leaving each
cell as its own independent voltage divider against its dedicated column
resistance):

```
I_j = sum_i  V_i / (R_cell_ij + R_col * (n - i))
```

Run the cross-check:

```
python3 examples/validate_analytic.py --rows 8 --cols 8 --r-line 10 --r-cell 1000 --v-in 0.7
python3 examples/validate_analytic.py --rows 32 --cols 32 --random --seed 3
```

On both uniform and randomized arrays up to 32x32 this comes back with
`max |ngspice - MNA|` around 1e-9 to 1e-10 A (against signals of order
1e-3 to 1e-2 A) — i.e. floating-point/solver noise, not a real
discrepancy — and the closed-form/MNA/ngspice triple for `combined_full`
matches to the same tolerance.

`crossbar/analytic.py` actually ships **two** MNA backends, kept side by
side rather than one replacing the other: `solve_analytic` (dense
`numpy.linalg.solve`) and `solve_analytic_sparse` (the identical stamps
assembled as a `scipy.sparse` matrix, solved with `spsolve`). ngspice and
the dense solver both become impractical well before 128x128 — ngspice
times out and the dense O(n³) solve exhausts memory — while the sparse
backend (the network only has a handful of nonzero entries per row) solves
a 128x128 array in a few seconds:

```
python3 examples/validate_analytic.py --rows 128 --cols 128 --r-line 1 --r-cell 10000 --v-in 0.7 --sparse --skip-ngspice
```

`--sparse` also cross-checks dense vs sparse against each other (agreeing
to ~1e-18, i.e. exactly) whenever the array is still small enough
(`rows*cols <= 4096`) to run the dense solver at all.

## Library usage

```python
from crossbar.topology import CrossbarConfig
from crossbar.compare import ideal_vmm, run_variant, compare_all
from crossbar.analytic import solve_analytic

cfg = CrossbarConfig(g=my_conductance_matrix, v_in=my_input_vector,
                      r_row=10.0, r_col=10.0)
ideal = ideal_vmm(cfg)          # numpy, no simulator needed (zero wire resistance)
measured = run_variant(cfg)     # runs ngspice, returns column currents
analytic = solve_analytic(cfg)  # same network, solved via MNA instead of ngspice
results = compare_all(cfg)      # baseline / wl_buffer_full / bl_star / combined_full
```

## Notes for extending this

- Cells are currently ideal linear resistors. Swapping in a nonlinear
  memristor model means replacing the `Rcell_*` lines in
  `crossbar/netlist.py` with a SPICE behavioral element (`B` source) or a
  compiled device model, and generalizing `.op` to `.dc`/`.tran` if the
  device is history-dependent.
- Schematics are generated directly from the regular grid topology
  (`crossbar/schematic.py`), not auto-laid-out from the netlist —
  general netlist-to-schematic layout is an unsolved problem for anything
  but the most regular circuits, so for irregular additions you may want
  to open the generated netlist in a tool like Xschem instead.
