"""Small helpers shared by every module. Nothing here imports torch or vllm."""
from __future__ import annotations

import json
import math
import numbers
import os
import signal
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

GIB = 2 ** 30
NAN = float("nan")


# ----------------------------------------------------------------------------
# subprocess / HTTP
# ----------------------------------------------------------------------------

def sh(cmd: list[str], timeout: float = 60) -> str:
    """stdout of a command, or "" if it fails or doesn't exist."""
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=timeout).strip()
    except Exception:  # noqa: BLE001
        return ""


def http_get(url: str, timeout: float = 5.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception:  # noqa: BLE001
        return 0, ""


def http_post_json(url: str, payload: Optional[dict] = None, timeout: float = 30.0) -> tuple[int, object]:
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            try:
                return r.status, json.loads(raw or b"null")
            except ValueError:
                return r.status, None
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:  # noqa: BLE001
        return 0, None


# ----------------------------------------------------------------------------
# host memory (unified-memory boxes: the page cache competes with CUDA)
# ----------------------------------------------------------------------------

def meminfo_gib() -> dict:
    """MemTotal / MemAvailable / page cache from /proc/meminfo, in GiB."""
    vals = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable", "Cached", "Buffers"):
                vals[k] = int(v.split()[0]) / 1048576
    except Exception:  # noqa: BLE001
        return {}
    return dict(mem_total_gib=round(vals.get("MemTotal", 0), 1),
                mem_available_gib=round(vals.get("MemAvailable", 0), 1),
                page_cache_gib=round(vals.get("Cached", 0) + vals.get("Buffers", 0), 1))


_DROP_WARNED = False


def drop_page_cache() -> bool:
    """sync + drop the page cache. On unified-memory parts (GB10, Jetson) the cache holds a
    second copy of the safetensors after every model load and CUDA can't reclaim it.
    Needs passwordless sudo for `tee /proc/sys/vm/drop_caches` (run.sh primes it)."""
    global _DROP_WARNED
    subprocess.run(["sync"], check=False)
    r = subprocess.run(["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"], input="3\n",
                       capture_output=True, text=True)
    if r.returncode != 0 and not _DROP_WARNED:
        _DROP_WARNED = True
        user = os.environ.get("USER", "<you>")
        print("  [mem] couldn't drop page cache (sudo needs a password). For unattended sweeps add:\n"
              f"        echo '{user} ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches' "
              "| sudo tee /etc/sudoers.d/drop-caches")
    return r.returncode == 0


# ----------------------------------------------------------------------------
# numbers / formatting
# ----------------------------------------------------------------------------

def is_num(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and not math.isnan(float(v)) \
        and not math.isinf(float(v))


def pctl(xs: list[float], q: float) -> float:
    """Linear-interpolated percentile (numpy's default), without numpy."""
    if not xs:
        return NAN
    s = sorted(xs)
    k = (len(s) - 1) * q / 100.0
    f = math.floor(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def pct_change(new, base) -> float:
    try:
        if not is_num(new) or not is_num(base) or base == 0:
            return NAN
        return (new - base) / base * 100.0
    except TypeError:
        return NAN


def fmt_cell(v, floatfmt: str = "{:,.1f}") -> str:
    if v is None:
        return "–"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "–"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, numbers.Integral):
        return f"{int(v):,}"
    if isinstance(v, numbers.Real):
        return floatfmt.format(float(v))
    return str(v)


def md_table(df, floatfmt: str = "{:,.1f}", index: bool = True, index_name: str = "") -> str:
    """pandas DataFrame -> GitHub markdown table (no tabulate dependency)."""
    if df is None or len(df) == 0:
        return "_no data_\n"
    cols = list(df.columns)
    headers = ([index_name or (df.index.name or "")] if index else []) + [str(c) for c in cols]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)] if index else []
        cells += [fmt_cell(row[c], floatfmt) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_json(path: Path, obj) -> None:
    """Atomic-enough JSON write (temp file + rename) so a crash never leaves half a file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default))
    tmp.replace(path)


def _json_default(o):
    if isinstance(o, float):
        return o
    if hasattr(o, "item"):          # numpy scalars
        return o.item()
    return str(o)


def exit_on_term() -> None:
    """`kill <pid>` / a closed terminal should still run the finally: blocks that stop the vLLM
    server (it runs in its own session, so it would otherwise outlive us and hold the GPU)."""
    def bye(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, bye)
    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:   # keep nohup's ignore for BG=1
        signal.signal(signal.SIGHUP, bye)
