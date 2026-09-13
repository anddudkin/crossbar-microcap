"""Compare VMM accuracy across compensation variants against the ideal (no
wire resistance) result."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from .analytic import solve_analytic, solve_analytic_sparse
from .netlist import generate_netlist
from .simulate import run_ngspice, read_column_currents
from .topology import CrossbarConfig


def ideal_vmm(cfg: CrossbarConfig) -> np.ndarray:
    """I_j = sum_i V_i * G_ij, i.e. the result with zero wire resistance."""
    return cfg.v_in @ cfg.g


def run_variant(cfg: CrossbarConfig) -> np.ndarray:
    netlist = generate_netlist(cfg)
    stdout = run_ngspice(netlist)
    currents = read_column_currents(stdout, cfg.n_cols)
    return np.array(currents)


#: Maps a backend name to a `cfg -> column currents` callable, so a script
#: that lets the user pick ngspice vs one of the two independent MNA
#: solvers from crossbar.analytic (dense/sparse — see that module for when
#: each one is practical) doesn't have to redefine this dispatch itself.
#: Originally declared ad hoc in examples/run_compare.py; centralized here
#: once a second script (examples/run_error_landscape.py) needed the same
#: thing, so the two/three copies couldn't drift apart.
BACKENDS: dict[str, Callable[[CrossbarConfig], np.ndarray]] = {
    "ngspice": run_variant,
    "dense": solve_analytic,
    "sparse": solve_analytic_sparse,
}


@dataclass
class VariantResult:
    label: str
    currents: np.ndarray
    rmse: float
    max_abs_error: float
    max_rel_error: float
    mean_rel_error: float


def compute_errors(ideal: np.ndarray, measured: np.ndarray) -> tuple[float, float, float, float]:
    """RMSE and max are single worst-case-flavoured summaries (RMSE is
    absolute, in amps, and dominated by whichever column is off by most in
    absolute terms; max_rel_error is the single worst column's relative
    error). mean_rel_error averages the relative error across every
    column instead, which is usually the more representative number for
    "how well does this compensate overall" rather than "how bad can it
    get on one column". Public (not just used internally by
    `evaluate_variant`) so scripts using a different backend than ngspice
    (e.g. examples/run_compare.py, which can also use the MNA solvers from
    crossbar.analytic) can score their own results the same way."""
    err = measured - ideal
    rel = np.abs(err) / np.abs(ideal)
    rmse = float(np.sqrt(np.mean(err**2)))
    max_abs = float(np.max(np.abs(err)))
    max_rel = float(np.max(rel))
    mean_rel = float(np.mean(rel))
    return rmse, max_abs, max_rel, mean_rel


def evaluate_variant(label: str, cfg: CrossbarConfig, ideal: np.ndarray) -> VariantResult:
    """Run `cfg` through ngspice and score it against a precomputed `ideal`
    VMM result. Exposed separately from `compare_all` so other scripts (e.g.
    examples/run_wl_buffer_interval_sweep.py) can build their own variant
    sets without duplicating the ngspice-run + error-metric plumbing."""
    currents = run_variant(cfg)
    rmse, max_abs, max_rel, mean_rel = compute_errors(ideal, currents)
    return VariantResult(
        label=label, currents=currents, rmse=rmse, max_abs_error=max_abs,
        max_rel_error=max_rel, mean_rel_error=mean_rel,
    )


#: The four compensation variants compare_all runs, as CrossbarConfig
#: overrides relative to a `base_cfg` that itself should have
#: buffer_interval=0, source_buffer=False, star_columns=False. Exposed as
#: its own function (not inlined in compare_all) so other scripts — e.g.
#: examples/run_compare.py, which lets the user pick a subset — can build
#: the same four variants without redefining them and risking drift.
#:
#: "Full WL buffering" means source_buffer=True and buffer_interval=1: a
#: buffer right at the row's source plus one before every crosspoint, which
#: fully cancels row IR-drop (see crossbar/topology.py). The
#: *partial/periodic* buffering variant (buffer_interval > 1, no source
#: buffer) is a separate, later research question — see
#: examples/run_wl_buffer_interval_sweep.py, which uses `evaluate_variant`
#: directly instead of this function.
def variant_configs(base_cfg: CrossbarConfig) -> dict[str, CrossbarConfig]:
    return {
        "baseline": replace(base_cfg, buffer_interval=0, source_buffer=False, star_columns=False),
        "wl_buffer_full": replace(base_cfg, buffer_interval=1, source_buffer=True, star_columns=False),
        "bl_star": replace(base_cfg, buffer_interval=0, source_buffer=False, star_columns=True),
        "combined_full": replace(base_cfg, buffer_interval=1, source_buffer=True, star_columns=True),
    }


def compare_all(base_cfg: CrossbarConfig) -> list[VariantResult]:
    """Run all four variants from `variant_configs(base_cfg)` against the
    ideal VMM result. See `variant_configs` for what each one means."""
    ideal = ideal_vmm(base_cfg)
    return [evaluate_variant(label, cfg, ideal) for label, cfg in variant_configs(base_cfg).items()]


def print_report(ideal: np.ndarray, results: list[VariantResult]) -> None:
    print(f"Ideal (R_line=0) column currents: {np.array2string(ideal, precision=6)}")
    print()
    header = (
        f"{'variant':<16}{'RMSE (A)':>14}{'max |err| (A)':>16}"
        f"{'mean rel err':>14}{'max rel err':>14}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.label:<16}{r.rmse:>14.3e}{r.max_abs_error:>16.3e}"
            f"{r.mean_rel_error:>14.2%}{r.max_rel_error:>14.2%}"
        )
