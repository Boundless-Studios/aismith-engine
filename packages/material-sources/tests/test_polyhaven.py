import hashlib
import pytest
from material_sources.polyhaven import lock_material, ingest, search

DATA = b"fixture map bytes"
INFO = {"type": 1, "dimensions": [2000, 1000], "name": "Brick floor", "authors": {"Artist": "Scan"}}
FILES = {k: {"2k": {"png": {"url": "https://dl.polyhaven.org/" + k + ".png", "size": len(DATA),
                                 "md5": hashlib.md5(DATA).hexdigest()}}} for k in ("Diffuse", "nor_gl", "Rough")}

def lock(**kw):
    return lock_material("brick", info=kw.pop("info", INFO), files=kw.pop("files", FILES), **kw)

def test_search_and_lock():
    assert search("brick", catalog={"brick": INFO})[0]["source"] == "Poly Haven"
    a = lock()
    assert a == lock() and a["physical_size_metres"] == [2, 1]
    assert a["maps"][1]["semantic"] == "normal_opengl"
    assert a["license"] == "CC0-1.0"
    assert lock(info={"type": 1})["physical_size_metres"] is None


def test_no_silent_resolution_or_map_fallback():
    with pytest.raises(ValueError): lock(resolution="4k")
    with pytest.raises(ValueError): lock(files={})
    with pytest.raises(ValueError): lock(info={"type": 2})


def test_deterministic_ingest_and_tamper_detection(tmp_path):
    a = lock()
    path = ingest(a, tmp_path, fetch=lambda url: iter([DATA]))
    assert ingest(a, tmp_path, fetch=lambda url: pytest.fail("must not download twice")) == path
    (path / a["maps"][0]["filename"]).write_bytes(b"changed")
    with pytest.raises(ValueError): ingest(a, tmp_path)


def test_corrupt_partial_download_leaves_no_bundle(tmp_path):
    a = lock()
    with pytest.raises(ValueError): ingest(a, tmp_path, fetch=lambda url: iter([b"bad"]))
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError): ingest(a, tmp_path, fetch=lambda url: iter([DATA]), max_map_bytes=1)


def test_tampered_lock_and_bad_host(tmp_path):
    a = lock()
    a["maps"][0]["url"] = "https://example.com/x"
    with pytest.raises(ValueError): ingest(a, tmp_path)
    files = {"Diffuse": {"2k": {"png": {"url": "http://127.0.0.1/x", "size": 1, "md5": "a"*32}}}}
    with pytest.raises(ValueError): lock(files=files)
