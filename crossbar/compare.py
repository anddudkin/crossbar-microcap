"""Compare VMM accuracy across compensation variants against the ideal (no
wire resistance) result."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

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


@dataclass
class VariantResult:
    label: str
    currents: np.ndarray
    rmse: float
    max_abs_error: float
    max_rel_error: float


def _errors(ideal: np.ndarray, measured: np.ndarray) -> tuple[float, float, float]:
    err = measured - ideal
    rmse = float(np.sqrt(np.mean(err**2)))
    max_abs = float(np.max(np.abs(err)))
    max_rel = float(np.max(np.abs(err) / np.abs(ideal)))
    return rmse, max_abs, max_rel


def compare_all(
    base_cfg: CrossbarConfig,
    buffer_interval: int,
) -> list[VariantResult]:
    """Run baseline, WL-buffer, BL-star and combined variants of `base_cfg`
    (base_cfg itself should have buffer_interval=0, star_columns=False) and
    compare each against the ideal VMM result."""
    ideal = ideal_vmm(base_cfg)

    variants = {
        "baseline": replace(base_cfg, buffer_interval=0, star_columns=False),
        "wl_buffer": replace(base_cfg, buffer_interval=buffer_interval, star_columns=False),
        "bl_star": replace(base_cfg, buffer_interval=0, star_columns=True),
        "combined": replace(base_cfg, buffer_interval=buffer_interval, star_columns=True),
    }

    results = []
    for label, cfg in variants.items():
        currents = run_variant(cfg)
        rmse, max_abs, max_rel = _errors(ideal, currents)
        results.append(
            VariantResult(
                label=label,
                currents=currents,
                rmse=rmse,
                max_abs_error=max_abs,
                max_rel_error=max_rel,
            )
        )
    return results


def print_report(ideal: np.ndarray, results: list[VariantResult]) -> None:
    print(f"Ideal (R_line=0) column currents: {np.array2string(ideal, precision=6)}")
    print()
    header = f"{'variant':<12}{'RMSE (A)':>14}{'max |err| (A)':>16}{'max rel err':>14}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.label:<12}{r.rmse:>14.3e}{r.max_abs_error:>16.3e}{r.max_rel_error:>14.2%}")
