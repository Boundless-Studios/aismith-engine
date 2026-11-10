"""Poly Haven metadata, explicit resolution lock and verified content ingestion."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import urllib.request
from urllib.parse import urlsplit

UA = "AISmith-MaterialSources/0.1 (Powered by Poly Haven)"
API = "https://api.polyhaven.com"
MAPS = {"Diffuse": ("base_color", "sRGB"), "nor_gl": ("normal_opengl", "non-color"),
        "Rough": ("roughness", "non-color"), "AO": ("occlusion", "non-color"),
        "Displacement": ("displacement", "non-color"), "Metal": ("metallic", "non-color")}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def slug(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9_]+", value):
        raise ValueError("invalid Poly Haven asset ID")
    return value


def check_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in ("dl.polyhaven.org", "cdn.polyhaven.com")
            or parsed.username or parsed.password or parsed.port not in (None, 443)):
        raise ValueError("unexpected Poly Haven download URL")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("unexpected redirect; refresh the source inventory")


def get_json(endpoint):
    request = urllib.request.Request(API + endpoint, headers={"User-Agent": UA})
    with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
        data = response.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("metadata response too large")
    return json.loads(data)


def search(query: str, *, catalog=None, limit=20):
    if not 1 <= limit <= 100:
        raise ValueError("limit must be 1..100")
    catalog = get_json("/assets?t=textures") if catalog is None else catalog
    terms = query.casefold().split()
    matches = []
    for asset_id, metadata in sorted(catalog.items()):
        text = " ".join([asset_id, metadata.get("name", ""), *metadata.get("tags", []), *metadata.get("categories", [])]).casefold()
        if all(term in text for term in terms):
            matches.append({"id": asset_id, "name": metadata.get("name", asset_id),
                            "tags": metadata.get("tags", []), "categories": metadata.get("categories", []),
                            "source": "Poly Haven", "credit": "Powered by Poly Haven"})
    return matches[:limit]


def lock_material(asset_id, *, resolution="2k", required=("Diffuse", "nor_gl", "Rough"),
                  optional=("AO", "Displacement", "Metal"), info=None, files=None):
    """No fallback/downscale: every required map must exist at exactly this resolution."""
    slug(asset_id)
    if not re.fullmatch(r"[1-9][0-9]*k", resolution):
        raise ValueError("invalid resolution")
    if not required or set(required) & set(optional) or any(k not in MAPS for k in (*required, *optional)):
        raise ValueError("unsupported or overlapping map selection")
    info = get_json("/info/" + asset_id) if info is None else info
    files = get_json("/files/" + asset_id) if files is None else files
    if info.get("type") != 1:
        raise ValueError("asset is not a material/texture")
    dimensions = info.get("dimensions")
    if dimensions is not None:
        if len(dimensions) < 2 or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in dimensions[:2]):
            raise ValueError("invalid physical dimensions")
        metres = [x / 1000 for x in dimensions[:2]]
    else:
        metres = None  # Never infer dimensions from texture resolution or a prose label.
    selected = []
    for kind in (*required, *optional):
        entry = files.get(kind, {}).get(resolution, {}).get("png")
        if not entry:
            if kind in required:
                raise ValueError(f"missing required {kind}/{resolution}/png")
            continue
        check_url(entry["url"])
        if not re.fullmatch(r"[0-9a-f]{32}", entry["md5"]) or type(entry["size"]) is not int or entry["size"] <= 0:
            raise ValueError("invalid map checksum or size")
        selected.append({"map": kind, "semantic": MAPS[kind][0], "color_space": MAPS[kind][1],
                         "url": entry["url"], "md5": entry["md5"], "size": entry["size"],
                         "filename": f"{asset_id}_{kind.lower()}_{resolution}.png"})
    result = {"schema_version": 1, "adapter_version": "1", "asset_id": asset_id,
              "name": info.get("name", asset_id), "source": "Poly Haven",
              "source_url": "https://polyhaven.com/a/" + asset_id,
              "license": "CC0-1.0", "license_url": "https://polyhaven.com/license",
              "api_terms_url": "https://github.com/Poly-Haven/Public-API/blob/master/ToS.md",
              "credit": "Powered by Poly Haven", "authors": info.get("authors", {}),
              "tags": info.get("tags", []), "categories": info.get("categories", []),
              "physical_size_metres": metres, "physical_scale_status": "known" if metres else "unmeasured",
              "resolution": resolution, "format": "png", "required": list(required), "optional": list(optional),
              "metadata_sha256": digest(info), "inventory_sha256": digest(files),
              "available_maps": {k: sorted(v) for k, v in files.items() if k in MAPS},
              "maps": selected}
    result["lock_sha256"] = digest(result)
    return result


def ingest(lock, destination: Path, *, fetch=None, max_map_bytes=512 * 1024 * 1024):
    """Content-addressed, all-or-nothing bundle. Fetch callback returns an iterator of bytes."""
    data = dict(lock)
    expected = data.pop("lock_sha256")
    if digest(data) != expected or data["schema_version"] != 1:
        raise ValueError("material lock changed")
    slug(data["asset_id"])
    if not data["maps"]:
        raise ValueError("empty material lock")
    names = [m["filename"] for m in data["maps"]]
    if len(set(names)) != len(names) or any(Path(n).name != n or n in (".", "..") for n in names):
        raise ValueError("invalid map filenames")
    for m in data["maps"]:
        check_url(m["url"])
        if type(m["size"]) is not int or not 0 < m["size"] <= max_map_bytes:
            raise ValueError("map exceeds ingestion size limit")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / expected
    # Existing bundles are never silently trusted or overwritten.
    if target.exists():
        manifest = json.loads((target / "manifest.json").read_text())
        if manifest["lock"] != lock:
            raise ValueError("existing bundle lock differs")
        if set(manifest["sha256"]) != set(names):
            raise ValueError("existing bundle inventory differs")
        for m in data["maps"]:
            payload = (target / m["filename"]).read_bytes()
            if (hashlib.sha256(payload).hexdigest() != manifest["sha256"][m["filename"]] or
                    hashlib.md5(payload).hexdigest() != m["md5"] or len(payload) != m["size"]):
                raise ValueError("existing material bytes changed")
        return target
    hashes = {}
    with tempfile.TemporaryDirectory(prefix=".polyhaven-", dir=destination) as tmp:
        stage = Path(tmp)
        for m in data["maps"]:
            md5, sha, size = hashlib.md5(), hashlib.sha256(), 0
            with (stage / m["filename"]).open("xb") as out:
                for chunk in (fetch or download)(m["url"]):
                    size += len(chunk)
                    if size > m["size"] or size > max_map_bytes:
                        raise ValueError("download exceeds locked size")
                    out.write(chunk)
                    md5.update(chunk)
                    sha.update(chunk)
            if size != m["size"] or md5.hexdigest() != m["md5"]:
                raise ValueError("download checksum/size mismatch")
            hashes[m["filename"]] = sha.hexdigest()
        (stage / "manifest.json").write_text(json.dumps({"lock": lock, "sha256": hashes}, indent=2) + "\n")
        # Directory rename commits the complete bundle; concurrent identical writer may win.
        try:
            stage.rename(target)
        except OSError:
            if target.exists():
                return ingest(lock, destination, fetch=fetch, max_map_bytes=max_map_bytes)
            raise
    return target


def download(url):
    check_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.build_opener(NoRedirect).open(req, timeout=60) as response:
        while chunk := response.read(1024 * 1024):
            yield chunk
