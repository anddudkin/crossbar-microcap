"""Run a generated netlist through ngspice (batch mode) and parse results."""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path


_OP_LINE_RE = re.compile(r"^\s*([a-zA-Z0-9_@().]+)\s*=\s*([-+0-9.eE]+)\s*$")

# On Windows, the official ngspice distribution ships two binaries: ngspice.exe
# (GUI-subsystem build, does not attach to a piped/captured stdout when run
# headless — batch-mode output comes back empty) and ngspice_con.exe (the
# console-subsystem build meant for exactly this kind of subprocess/batch
# use). Plain "ngspice" resolves to the former on Windows, so default to the
# console build there; Linux packages only provide a plain "ngspice" binary.
_DEFAULT_NGSPICE_BIN = "ngspice_con" if sys.platform == "win32" else "ngspice"


def run_ngspice(netlist: str, ngspice_bin: str = _DEFAULT_NGSPICE_BIN) -> str:
    """Write `netlist` to a temp file, run ngspice -b on it, return raw stdout."""
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(netlist)
        path = Path(f.name)
    try:
        result = subprocess.run(
            [ngspice_bin, "-b", str(path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ngspice exited with {result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        return result.stdout
    finally:
        path.unlink(missing_ok=True)


def read_column_currents(stdout: str, n_cols: int) -> list[float]:
    """Parse `i(Vsense_j) = <value>` lines from ngspice's `.op` print output.

    ngspice reports the branch current of `Vsense_j` (from + to - through the
    source) as the current flowing into the array's virtual-ground node from
    the column, which is exactly the physical VMM output current for that
    column (verified against the zero-resistance ideal case). No sign flip
    needed.
    """
    values: dict[int, float] = {}
    for raw_line in stdout.splitlines():
        m = _OP_LINE_RE.match(raw_line)
        if not m:
            continue
        name, val = m.group(1).lower(), m.group(2)
        cell = re.match(r"i\(vsense_(\d+)\)", name)
        if cell:
            values[int(cell.group(1))] = float(val)
    missing = [j for j in range(n_cols) if j not in values]
    if missing:
        raise ValueError(
            f"could not find column current(s) for columns {missing} in ngspice output:\n{stdout}"
        )
    return [values[j] for j in range(n_cols)]
