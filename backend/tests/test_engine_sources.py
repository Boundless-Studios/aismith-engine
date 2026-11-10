"""The sources lock: every vendored submodule and install-time clone is pinned."""
from __future__ import annotations

import configparser
import importlib.util
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.engine import sources

ROOT = sources.ROOT


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def _gitlinks() -> dict[str, str]:
    """Paths recorded in the index as submodules (mode 160000) -> commit."""
    listing = _git("ls-files", "--stage", "--", "vendor", cwd=ROOT)
    links = {}
    for line in listing.splitlines():
        meta, path = line.split("\t", 1)
        mode, commit, _stage = meta.split()
        if mode == "160000":
            links[path] = commit
    return links


def _gitmodules() -> dict[str, str]:
    parser = configparser.ConfigParser()
    parser.read(ROOT / ".gitmodules")
    return {parser[section]["path"]: parser[section]["url"] for section in parser.sections()}


def test_every_gitlink_is_declared_and_matches_the_lock():
    links = _gitlinks()
    assert links, "expected vendored submodules"
    modules = _gitmodules()
    locked = {s.vendor_path: s for s in sources.load_sources().values() if s.vendor_path}
    assert set(links) == set(modules) == set(locked)
    for path, commit in links.items():
        assert locked[path].commit == commit, path
        assert modules[path] == locked[path].repository, path


def test_every_source_has_a_licence_and_full_commit():
    for pinned in sources.load_sources().values():
        assert pinned.licence and pinned.repository.startswith("https://github.com/")
        assert len(pinned.commit) == 40


def test_comfyui_runtime_is_pinned_not_a_branch():
    comfyui = sources.source("comfyui")
    assert comfyui.vendor_path is None
    assert comfyui.ref and comfyui.ref.startswith("v")


def test_native_installer_reads_its_pins_from_the_lock():
    path = ROOT / "backend" / "workers" / "install_trellis2_native.py"
    spec = importlib.util.spec_from_file_location("install_trellis2_native_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, repository, commit in (
        ("cumesh", module.CUMESH_REPOSITORY, module.CUMESH_COMMIT),
        ("cumesh-api", module.CUMESH_API_REPOSITORY, module.CUMESH_API_COMMIT),
        ("trellis2", module.TRELLIS_REPOSITORY, module.TRELLIS_COMMIT),
    ):
        pinned = sources.source(name)
        assert (repository, commit) == (pinned.repository, pinned.commit)


def test_lock_rejects_an_abbreviated_commit(tmp_path):
    lock = tmp_path / "sources.lock.json"
    lock.write_text(json.dumps({"schema_version": 1, "sources": [
        {"name": "x", "repository": "https://github.com/a/b.git", "commit": "abc123", "licence": "MIT", "used_by": "test"},
    ]}))
    with pytest.raises(sources.SourceError, match="40-character"):
        sources.load_sources(lock)


@pytest.fixture
def upstream(tmp_path):
    """A local repository with two commits, plus a lock pinning the first."""
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    for message in ("first", "second"):
        (repo / "file.txt").write_text(message)
        _git("add", "file.txt", cwd=repo)
        _git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", message, cwd=repo)
    first = _git("rev-parse", "HEAD~1", cwd=repo)
    lock = tmp_path / "sources.lock.json"
    lock.write_text(json.dumps({"schema_version": 1, "sources": [
        {"name": "dep", "repository": repo.as_uri(), "commit": first, "licence": "MIT",
         "used_by": "test", "vendor_path": "vendor/dep"},
    ]}))
    return repo, first, lock


def test_clone_pinned_checks_out_the_locked_commit(upstream, tmp_path):
    _repo, first, lock = upstream
    destination = tmp_path / "out" / "dep"
    sources.clone_pinned("dep", destination, lock)
    assert _git("rev-parse", "HEAD", cwd=destination) == first
    assert (destination / "file.txt").read_text() == "first"
    with pytest.raises(sources.SourceError, match="already exists"):
        sources.clone_pinned("dep", destination, lock)


def test_require_vendored_explains_an_uninitialised_submodule(upstream, tmp_path):
    _repo, _first, lock = upstream
    (tmp_path / "vendor" / "dep").mkdir(parents=True)
    with pytest.raises(sources.SourceError, match="git submodule update --init vendor/dep"):
        sources.require_vendored("dep", root=tmp_path, lock_path=lock)


def test_require_vendored_rejects_a_checkout_at_another_commit(upstream, tmp_path):
    repo, first, lock = upstream
    checkout = tmp_path / "vendor" / "dep"
    _git("clone", "-q", repo.as_uri(), str(checkout), cwd=tmp_path)
    with pytest.raises(sources.SourceError, match="lock pins"):
        sources.require_vendored("dep", root=tmp_path, lock_path=lock)
    _git("checkout", "-q", "--detach", first, cwd=checkout)
    assert sources.require_vendored("dep", root=tmp_path, lock_path=lock) == checkout


def test_refiner_install_fails_clearly_without_the_submodule(tmp_path):
    from backend.app import main

    real = sources.require_vendored
    state = main.DownloadState(id="test-refiner", model="trellis-refiner")
    with patch.object(main, "DOWNLOADS_DIR", tmp_path), \
         patch.object(main.native, "trellis_refiner_dir", return_value=tmp_path / "runtime"), \
         patch.object(main.engine_sources, "require_vendored", lambda name: real(name, root=tmp_path)):
        (tmp_path / "vendor" / "ComfyUI-Trellis2").mkdir(parents=True)
        main._run_trellis_refiner_download(state)
    assert state.status == "failed"
    assert "git submodule update --init vendor/ComfyUI-Trellis2" in (state.error or "")
