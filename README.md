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

- **`wl_buffer`** — every `buffer_interval`-th crosspoint along a row is
  fed by an ideal unity-gain buffer referenced to the row's own source
  node, instead of through a passive `R_row` segment. This re-drives the
  row to the exact intended `V_i`, resetting accumulated IR-drop
  periodically (the repeater idea from `base_r_line_10_WL_invertor.cir`).
- **`bl_star`** — each crosspoint gets its own dedicated wire straight to
  the column's virtual ground (resistance proportional to its distance
  from the bottom), instead of a shared chain of `R_col` segments. This
  removes cross-cell coupling through shared bit-line resistance.
- **`combined`** — both of the above together.
- **`baseline`** — plain resistive crossbar, no compensation, for reference.

## Requirements

```
sudo apt-get install ngspice
pip install -r requirements.txt
```

## Usage

```
python3 examples/run_demo.py --rows 8 --cols 8 --r-line 10 --buffer-interval 2
```

This will:
1. Build a random `rows x cols` conductance matrix and input vector.
2. Run all four variants through ngspice and compare each against the
   ideal (zero-resistance) VMM result — RMSE, max absolute and max
   relative column error (`out/vmm_comparison.csv`, printed to stdout).
3. Save a bar-chart comparison (`out/error_comparison.png`).
4. Render illustrative schematics of a small `NxN` subset for all four
   variants (`out/schematics/*.svg` and `.png`) — vector figures suitable
   for a paper or patent, drawn directly from the array topology rather
   than auto-laid-out from the netlist.

## Library usage

```python
from crossbar.topology import CrossbarConfig
from crossbar.compare import ideal_vmm, run_variant, compare_all

cfg = CrossbarConfig(g=my_conductance_matrix, v_in=my_input_vector,
                      r_row=10.0, r_col=10.0)
ideal = ideal_vmm(cfg)                       # numpy, no simulator needed
measured = run_variant(cfg)                  # runs ngspice, returns column currents
results = compare_all(cfg, buffer_interval=2)  # baseline/wl_buffer/bl_star/combined
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
