"""The licence gate (scripts/check_licences.py): every lock entry and every
copyleft, non-commercial or custom-licensed package is named in NOTICE."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("check_licences", ROOT / "scripts" / "check_licences.py")
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


@pytest.mark.parametrize(("expression", "expected"), [
    ("MIT", False),
    ("Apache-2.0", False),
    ("MPL-2.0", False),
    ("GPL-3.0", True),
    ("GNU General Public License v3 (GPLv3)", True),
    ("LGPL-3.0-or-later", True),
    ("AGPL-3.0", True),
    ("(MIT OR GPL-3.0-or-later)", False),
    ("Apache-2.0 AND LGPL-3.0-or-later", True),
    ("LicenseRef-MeshLib", True),
    ("CC-BY-NC-4.0", True),
    ("NVIDIA Source Code License (non-commercial)", True),
    ("GPL-3.0-only AND (MIT OR Apache-2.0)", True),
    ("(MIT OR Apache-2.0) AND GPL-3.0-only", True),
    ("MIT AND (GPL-2.0-only OR Apache-2.0)", False),
    ("GPL-2.0-or-later WITH Classpath-exception-2.0", True),
    ("MIT/Apache-2.0", False),
    ("", False),
])
def test_flags_copyleft_non_commercial_and_custom_licences(expression, expected):
    assert gate.flagged(expression) is expected


def test_names_match_whole_words_and_platform_families():
    notice = "pymeshfix is optional. sharp (LGPL-3.0 libvips) is build-time only."
    assert gate.named_in(notice, "pymeshfix")
    assert gate.named_in(notice, "PyMeshFix")
    assert not gate.named_in(notice, "meshfix")
    assert gate.named_in(notice, "@img/sharp-libvips-linux-x64")
    assert not gate.named_in(notice, "@img/other-libvips")


def _fake_repo(tmp_path: Path, notice: str) -> Path:
    (tmp_path / "vendor").mkdir()
    for name in ("sources.lock.json", "weights.lock.json"):
        shutil.copy(ROOT / "vendor" / name, tmp_path / "vendor" / name)
    (tmp_path / "NOTICE").write_text(notice, encoding="utf-8")
    return tmp_path


def test_every_lock_entry_is_named_in_notice():
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert gate.check_locks(ROOT, notice) == []


def test_a_lock_entry_missing_from_notice_fails(tmp_path):
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8").replace("comfyanonymous/ComfyUI", "ComfyUI")
    root = _fake_repo(tmp_path, notice)
    problems = gate.check_locks(root, notice)
    assert any("comfyui" in problem and "not named in NOTICE" in problem for problem in problems)


def test_a_lock_entry_without_a_licence_fails(tmp_path):
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    root = _fake_repo(tmp_path, notice)
    lock = json.loads((root / "vendor" / "weights.lock.json").read_text())
    lock["weights"][0]["licence"] = ""
    (root / "vendor" / "weights.lock.json").write_text(json.dumps(lock))
    assert any("has no licence" in problem for problem in gate.check_locks(root, notice))


def test_an_unannounced_copyleft_npm_package_fails(tmp_path):
    root = _fake_repo(tmp_path, "nothing named here")
    for lock in ("frontend/package-lock.json", "vendor/mesh2motion-app/package-lock.json"):
        path = root / lock
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"packages": {
            "": {"name": "project"},
            "node_modules/left-pad": {"license": "MIT"},
            "node_modules/dual": {"license": "(MIT OR GPL-3.0-or-later)"},
            "node_modules/copyleft-thing": {"license": "GPL-3.0"},
        }}))
    problems, unknown = gate.check_npm(root, "nothing named here")
    assert len(problems) == 2 and all("copyleft-thing" in problem for problem in problems)
    assert unknown == []
