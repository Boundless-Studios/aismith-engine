"""A process's descendants, wherever their session or process group.

Native workers that call setsid leave their parent's process group, so a
group kill cannot reach them. These helpers find them by parent pid instead.
Linux reads /proc (standard library only); elsewhere `ps` is used, which is
enough for development on macOS.

Each pid comes with a start token (the kernel start time on Linux), so a pid
the OS has since reused for an unrelated process is never signalled.
"""
from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

PROC = Path("/proc")


def _table() -> dict[int, tuple[int, str]]:
    """pid -> (parent pid, start token) for every visible process."""
    table: dict[int, tuple[int, str]] = {}
    if PROC.is_dir() and (PROC / "self" / "stat").is_file():
        for entry in PROC.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "stat").read_text()
            except OSError:
                continue
            # "pid (comm) state ppid ... starttime(22) ..."; comm may contain spaces or ")".
            fields = raw[raw.rfind(")") + 2:].split()
            if len(fields) > 19 and fields[0] != "Z":
                table[int(entry.name)] = (int(fields[1]), fields[19])
        return table
    try:
        output = subprocess.run(["ps", "-axo", "pid=,ppid=,stat=,lstart="], capture_output=True, text=True,
                                timeout=10, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return table
    for line in output.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4 and parts[0].isdigit() and parts[1].isdigit() and not parts[2].startswith("Z"):
            table[int(parts[0])] = (int(parts[1]), parts[3].strip())
    return table


def descendants(root: int) -> dict[int, str]:
    """Every live descendant of `root` (not `root` itself): pid -> start token."""
    table = _table()
    children: dict[int, list[int]] = {}
    for pid, (ppid, _token) in table.items():
        children.setdefault(ppid, []).append(pid)
    found: dict[int, str] = {}
    stack = list(children.get(root, ()))
    while stack:
        pid = stack.pop()
        if pid in found or pid == root:
            continue
        found[pid] = table[pid][1]
        stack.extend(children.get(pid, ()))
    return found


def alive(pids: dict[int, str]) -> dict[int, str]:
    """The subset of `pids` still running as the same process (zombies excluded)."""
    table = _table()
    return {pid: token for pid, token in pids.items() if pid in table and table[pid][1] == token}


def signal_all(pids: dict[int, str], sig: int = signal.SIGKILL) -> None:
    """Signal each pid that is still the process it was when recorded."""
    for pid in alive(pids):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def reap(pids: dict[int, str]) -> None:
    """Reap any of `pids` that became our children (we are PID 1 or a subreaper)."""
    for pid in pids:
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
