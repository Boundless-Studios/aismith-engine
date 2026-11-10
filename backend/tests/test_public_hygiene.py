"""Nothing private comes back into the public tree.

Every file Git tracks is scanned for what this repository must never carry:
- email addresses outside the documentation domains;
- private, carrier-grade NAT (Tailscale) and link-local addresses;
- tailnet host names;
- personal home directories and drive mounts;
- private tracker ids and links to private repositories;
- references to agent handoff logs and internal plans;
- a short list of private machine, account and repository names. Only their
  sha256 hashes are listed here, so the list itself stays private.

Findings name the rule, file and line, never the matched text. Upstream's
vendored C++ tree is skipped; the secret scan (gitleaks) still covers it.

In the private source repository, `export/public-snapshot.toml` exists and
lists what the public snapshot leaves out. Those paths are skipped here, and
the exporter's own, stricter scan covers the rest.
"""
from __future__ import annotations

import fnmatch
import hashlib
import ipaddress
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PRIVATE_MANIFEST = ROOT / "export" / "public-snapshot.toml"
TEXT_SAMPLE = 8192
SKIP = ("vendor/trellis2cpp/",)

ALLOWED_EMAIL_DOMAINS = ("example.com", "example.org", "example.net", "example", "invalid", "test", "localhost", "local")
ALLOWED_EMAILS = ("git@github.com",)
# Private addresses tests use on purpose: the engine runner's bind policy
# refuses them without --allow-private (test_engine_serve.py).
ALLOWED_ADDRESSES = (b"10.0.0.5", b"172.17.0.2")
PRIVATE_NETWORKS = tuple(ipaddress.ip_network(n) for n in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",  # RFC 1918
    "100.64.0.0/10",  # carrier-grade NAT, which Tailscale uses
    "169.254.0.0/16",  # link-local
))

EMAIL = re.compile(rb"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])")
IPV4 = re.compile(rb"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
CIDR_SUFFIX = re.compile(rb"/(\d{1,2})(?![\d/.])")


def _is_network_literal(address: bytes, suffix: re.Match[bytes] | None) -> bool:
    """`10.0.0.0/8` names a range; a host with a prefix (`192.0.2.20/24`) and
    a URL path (`http://192.0.2.20/80/status`) still name a machine."""
    if suffix is None:
        return False
    try:
        ipaddress.ip_network(f"{address.decode()}/{suffix.group(1).decode()}", strict=True)
    except ValueError:
        return False
    return True
TOKEN = re.compile(rb"[A-Za-z0-9][A-Za-z0-9_-]*")
PATTERNS = {
    "tailscale-name": re.compile(rb"\b[a-z0-9-]+\.[a-z0-9-]+\.ts\.net\b", re.IGNORECASE),
    "home-directory": re.compile(rb"/(home|Users)/(?!you\b|user\b|me\b|runner\b|worker\b|your-user\b)[a-z][a-z0-9_-]+/"),
    "drive-mount": re.compile(rb"/mnt/[a-z]/"),
    "windows-user-path": re.compile(rb"(?i)[a-z]:\\Users\\(?!public\b)[a-z]"),
    "tracker-id": re.compile(rb"\bBOU-\d+\b"),
    "private-repository-link": re.compile(rb"github\.com/Boundless-Studios/AISmith-3D\b"),
    "handoff-log": re.compile(rb"\b(HANDOFF|TODO)-[A-Z0-9-]+\.md\b"),
    "internal-plan": re.compile(rb"\bdocs/(plans|superpowers)/"),
}
# sha256 of private machine, tailnet, account and repository names, matched
# against whole tokens (case-insensitive).
PRIVATE_TOKEN_SHA256 = frozenset({
    "e6031f6752cffc4d758955810f25488b13092bf0d4c1062a0849b88b64228b3b",
    "b927ddcf2a6f368eb0f4a57fddac7317668319872fed6e77412643b7e418a7c2",
    "3fc8c3de139c763ec14c3750ff07d12fce580318d0935dc80cb6166560e648f6",
    "44530edcd79410d098911b56f1c251a3e020d82a7f1d1a69c4a0673e6b15fd76",
    "79012ea27cc5f7222c4532c0ae1866fe17fb21843139b5d3daf6a8ec8b903dfd",
})
PRIVATE_TOKEN_LENGTHS = frozenset({8, 10, 12, 14, 18})


def _matches(path: str, rule: str) -> bool:
    if rule.endswith("/"):
        return path.startswith(rule)
    return path == rule or fnmatch.fnmatchcase(path, rule)


def _private_exclusions() -> tuple[str, ...]:
    if not PRIVATE_MANIFEST.is_file():
        return ()
    return tuple(tomllib.loads(PRIVATE_MANIFEST.read_text(encoding="utf-8")).get("exclude", ()))


def _tracked_files() -> list[str]:
    output = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"], check=True, capture_output=True
    ).stdout
    return sorted(name.decode() for name in output.split(b"\0") if name)


def _line(data: bytes, offset: int) -> int:
    return data.count(b"\n", 0, offset) + 1


def scan(files: dict[str, bytes]) -> list[dict]:
    """Findings by rule, file and line for each text file."""
    findings: list[dict] = []
    for path, data in sorted(files.items()):
        if b"\0" in data[:TEXT_SAMPLE]:
            continue

        def report(rule: str, offset: int) -> None:
            findings.append({"rule": rule, "file": path, "line": _line(data, offset)})

        for match in EMAIL.finditer(data):
            email = match.group().decode(errors="replace").lower()
            domain = email.rsplit("@", 1)[1]
            if email in ALLOWED_EMAILS or any(domain == d or domain.endswith("." + d) for d in ALLOWED_EMAIL_DOMAINS):
                continue
            report("email", match.start())
        for match in IPV4.finditer(data):
            if match.group() in ALLOWED_ADDRESSES:
                continue
            if _is_network_literal(match.group(), CIDR_SUFFIX.match(data, match.end())):
                continue  # a network range such as 10.0.0.0/8, not a machine
            try:
                address = ipaddress.ip_address(match.group().decode())
            except ValueError:
                continue
            if any(address in network for network in PRIVATE_NETWORKS):
                report("private-address", match.start())
        for rule, pattern in PATTERNS.items():
            for match in pattern.finditer(data):
                report(rule, match.start())
        for match in TOKEN.finditer(data):
            token = match.group()
            if len(token) in PRIVATE_TOKEN_LENGTHS and hashlib.sha256(token.lower()).hexdigest() in PRIVATE_TOKEN_SHA256:
                report("private-name", match.start())
    return findings


def test_the_scan_reports_rule_file_and_line_but_never_the_value():
    # Each sample is split across two literals so this file itself stays clean.
    files = {
        "a.md": b"ok\nwrite to someone" b"@gmail.com\n",
        "b.py": b"x = 'dev@example.com'\ny = 'git@github.com'\n",
        "c.txt": b"tailnet 100.101." b"102.103 and box.tail1234.ts" b".net\n"
                 b"lan 192.168." b"1.20 local 127.0.0.1 doc 192.0.2.10 range 10.0." b"0.0/8\n"
                 b"host 192.168." b"1.20/24 url http://10.1." b"2.3/80/status\n",
        "d.sh": b"ROOT=/home/" b"someone/work\nDATA=/mnt/" b"k/models\nok=/home/user/x\n",
        "e.md": b"see HANDOFF-" b"RIGGING.md and BOU-" b"1234\n",
        "f.png": b"\x89PNG\x00someone" b"@gmail.com",
    }
    findings = scan(files)
    assert findings == [
        {"rule": "email", "file": "a.md", "line": 2},
        {"rule": "private-address", "file": "c.txt", "line": 1},
        {"rule": "private-address", "file": "c.txt", "line": 2},
        {"rule": "private-address", "file": "c.txt", "line": 3},
        {"rule": "private-address", "file": "c.txt", "line": 3},
        {"rule": "tailscale-name", "file": "c.txt", "line": 1},
        {"rule": "home-directory", "file": "d.sh", "line": 1},
        {"rule": "drive-mount", "file": "d.sh", "line": 2},
        {"rule": "tracker-id", "file": "e.md", "line": 1},
        {"rule": "handoff-log", "file": "e.md", "line": 1},
    ]
    assert b"gmail" not in repr(findings).encode()


def test_private_names_are_matched_by_hash_as_whole_tokens():
    # A token whose hash is listed is found; the same letters inside a longer
    # token are not.
    name = b"magic-video-" b"editor"
    assert hashlib.sha256(name).hexdigest() in PRIVATE_TOKEN_SHA256
    assert scan({"x.md": b"see " + name + b" for details\n"}) == [
        {"rule": "private-name", "file": "x.md", "line": 1},
    ]
    assert scan({"x.md": b"see " + name + b"-fork\n"}) == []


@pytest.mark.skipif(
    shutil.which("git") is None
    or subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True).returncode != 0,
    reason="needs this repository's git checkout",
)
def test_the_tracked_tree_carries_nothing_private():
    exclusions = _private_exclusions() + SKIP
    files = {}
    for path in _tracked_files():
        if any(_matches(path, rule) for rule in exclusions):
            continue
        full = ROOT / path
        if full.is_file() and not full.is_symlink():
            files[path] = full.read_bytes()
    findings = scan(files)
    assert findings == [], "\n".join(f"{f['rule']}: {f['file']}:{f['line']}" for f in findings)
