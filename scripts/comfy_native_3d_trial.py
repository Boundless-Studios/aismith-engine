"""Headless A/B runner for ComfyUI 0.34's native TRELLIS.2 / Pixal3D
image-to-3D workflow ("3d_pixal3d_trellis2_image_to_model"), so its output
can be compared against our own pipeline on the same input image.

No GPU or ComfyUI is available in this worktree; this script is written to
run on a remote box that has both.

Usage:
    python3 scripts/comfy_native_3d_trial.py \\
        --server http://127.0.0.1:8288 --image path/to/input.png \\
        --mode pixal3d --resolution 1024 --out out/comfy_trial

Mode switch
-----------
The official template picks between the Pixal3D and TRELLIS.2 branches with
three `ComfySwitchNode`s, each fed by the same `PrimitiveBoolean` node
(titled "Boolean (Switch to Trellis2)" in the template, default value
`false`). The template also carries a `MarkdownNote` (id 317 in the
template this was written against) whose text reads:

    Set to `false` to use Pixal3D
    Set to `true` to use Trellis.2

This script does not hardcode that node id: after converting the UI graph
to an API prompt, it walks every `ComfySwitchNode`'s `switch` input back to
its source node (through the `PrimitiveBoolean`s that actually gate mode
selection), then infers which literal value means which mode by scanning
every `Note`/`MarkdownNote` node's text for the
"Set to `<bool>` to use <ModeName>" phrasing ComfyUI's own template
generator uses. If no such note is found (e.g. a future template revision
drops it), it falls back to the gating boolean node's `title`: a title
containing "trellis" implies `true` selects TRELLIS.2, everything else
(including no title) means Pixal3D is the `false` branch and TRELLIS.2 is
`true`, mirroring the template's own convention.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.app.comfy_graph import find_nodes, ui_workflow_to_api_prompt  # noqa: E402

DEFAULT_TEMPLATE_URL = (
    "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/"
    "templates/3d_pixal3d_trellis2_image_to_model.json"
)
MODE_KEYWORDS = {"pixal3d": "pixal3d", "trellis2": "trellis"}
MESH_EXTENSIONS = (".glb", ".obj", ".ply")
POLL_INTERVAL_SECONDS = 3


def load_template(template_path: str | None) -> dict:
    if template_path:
        return json.loads(Path(template_path).read_text())
    resp = requests.get(DEFAULT_TEMPLATE_URL, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_object_info(server: str) -> dict:
    resp = requests.get(f"{server}/object_info", timeout=60)
    resp.raise_for_status()
    return resp.json()


def upload_image(server: str, image_path: Path) -> str:
    with open(image_path, "rb") as fh:
        files = {"image": (image_path.name, fh)}
        resp = requests.post(f"{server}/upload/image", files=files, data={"overwrite": "true"}, timeout=120)
    resp.raise_for_status()
    return resp.json()["name"]


def set_loadimage_filenames(prompt: dict[str, dict], filename: str) -> None:
    for node_id in find_nodes(prompt, "LoadImage"):
        prompt[node_id]["inputs"]["image"] = filename


def _switch_gating_boolean_ids(prompt: dict[str, dict]) -> set[str]:
    boolean_ids: set[str] = set()
    for switch_id in find_nodes(prompt, "ComfySwitchNode"):
        switch_input = prompt[switch_id]["inputs"].get("switch")
        if isinstance(switch_input, list):
            boolean_ids.add(switch_input[0])
    return {node_id for node_id in boolean_ids if prompt.get(node_id, {}).get("class_type") == "PrimitiveBoolean"}


def _mode_from_note_text(workflow: dict, requested_mode: str) -> dict[str, bool] | None:
    pattern = re.compile(r"set\s+to\s+`?(true|false)`?\s+to\s+use\s+([a-z0-9_.]+)", re.IGNORECASE)
    mapping: dict[str, bool] = {}
    for node in workflow["nodes"]:
        if node.get("type") not in ("Note", "MarkdownNote"):
            continue
        text = " ".join(str(v) for v in (node.get("widgets_values") or []))
        for bool_str, mode_name in pattern.findall(text):
            for mode, keyword in MODE_KEYWORDS.items():
                if keyword in mode_name.lower():
                    mapping[mode] = bool_str.lower() == "true"
    if requested_mode in mapping:
        return mapping
    return None


def _mode_from_boolean_title(workflow: dict, boolean_id: str) -> dict[str, bool]:
    node = next((n for n in workflow["nodes"] if str(n["id"]) == boolean_id), None)
    title = ((node.get("title") if node else "") or "").lower()
    if "pixal3d" in title and "trellis" not in title:
        return {"trellis2": False, "pixal3d": True}
    return {"trellis2": True, "pixal3d": False}


def apply_mode(prompt: dict[str, dict], workflow: dict, mode: str) -> list[str]:
    boolean_ids = _switch_gating_boolean_ids(prompt)
    if not boolean_ids:
        raise RuntimeError("no PrimitiveBoolean gating a ComfySwitchNode was found in the converted graph")

    mapping = _mode_from_note_text(workflow, mode)
    applied = []
    for boolean_id in sorted(boolean_ids):
        node_mapping = mapping if mapping is not None else _mode_from_boolean_title(workflow, boolean_id)
        value = node_mapping[mode]
        prompt[boolean_id]["inputs"]["value"] = value
        applied.append(boolean_id)
    return applied


def set_resolution(prompt: dict[str, dict], resolution: int) -> None:
    for node_id in find_nodes(prompt, "Trellis2UpsampleStage"):
        prompt[node_id]["inputs"]["target_resolution"] = str(resolution)


def set_seed(prompt: dict[str, dict], seed: int) -> None:
    for node in prompt.values():
        for key in ("seed", "noise_seed"):
            if key in node["inputs"] and not isinstance(node["inputs"][key], list):
                node["inputs"][key] = seed


def submit_prompt(server: str, prompt: dict[str, dict], client_id: str) -> str:
    resp = requests.post(
        f"{server}/prompt", json={"prompt": prompt, "client_id": client_id}, timeout=60
    )
    if resp.status_code != 200:
        raise RuntimeError(f"POST /prompt failed ({resp.status_code}): {resp.text}")
    body = resp.json()
    node_errors = body.get("node_errors") or {}
    if node_errors:
        # ComfyUI validates per-branch: a node that fails validation (and
        # everything downstream of it) is silently pruned from execution
        # rather than rejecting the whole /prompt call, so a 200 here does
        # NOT mean the graph we asked for actually ran. Fail loud now
        # instead of discovering later that only an unrelated, unaffected
        # branch (e.g. the 2D preview nodes) executed.
        print("ComfyUI rejected part of the graph at validation (node_errors):", file=sys.stderr)
        print(json.dumps(node_errors, indent=2), file=sys.stderr)
        sys.exit(2)
    return body["prompt_id"]


def _find_error_message(messages: list) -> tuple[str, str] | None:
    for message in messages:
        if not isinstance(message, list) or len(message) != 2:
            continue
        event, data = message
        if not isinstance(data, dict):
            continue
        if "error" in str(event).lower():
            return str(data.get("node_id")), str(data.get("exception_message") or data)
        for value in data.values():
            if isinstance(value, str) and "ignored" in value.lower():
                return str(data.get("node_id")), value
    return None


def poll_history(server: str, prompt_id: str, timeout: int) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        resp = requests.get(f"{server}/history/{prompt_id}", timeout=30)
        resp.raise_for_status()
        history = resp.json()
        entry = history.get(prompt_id)
        if entry is not None:
            status = entry.get("status", {})
            messages = status.get("messages", [])
            error = _find_error_message(messages)
            if error is not None:
                node_id, message = error
                print(f"ComfyUI reported a node error/ignored output at node {node_id}: {message}", file=sys.stderr)
                sys.exit(2)
            if status.get("completed"):
                return entry
        if time.monotonic() >= deadline:
            raise TimeoutError(f"prompt {prompt_id} did not complete within {timeout}s")
        time.sleep(POLL_INTERVAL_SECONDS)


def verify_3d_branch_executed(prompt: dict[str, dict], history_entry: dict) -> list[str]:
    """Returns a list of problems, empty if the 3D-output branch actually ran.

    A prompt can complete "successfully" (status.completed True, no
    execution_error message) while the Save3DAdvanced/MeshToFile3D branch
    was silently pruned upstream at validation and only an unrelated part of
    the graph (e.g. the 2D preview nodes) executed. Detect that by checking
    the branch's own node ids actually produced outputs in the history.
    """
    problems = []
    outputs = history_entry.get("outputs", {})
    target_ids = find_nodes(prompt, "Save3DAdvanced") + find_nodes(prompt, "MeshToFile3D")
    for node_id in target_ids:
        if node_id not in outputs or not outputs[node_id]:
            class_type = prompt[node_id]["class_type"]
            problems.append(f"{class_type} node {node_id} produced no output (likely pruned at validation)")
    return problems


def _download(server: str, filename: str, subfolder: str, file_type: str, dest_dir: Path) -> Path:
    params = {"filename": filename, "subfolder": subfolder, "type": file_type}
    resp = requests.get(f"{server}/view", params=params, timeout=120, stream=True)
    resp.raise_for_status()
    dest = dest_dir / filename
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            fh.write(chunk)
    return dest


def collect_outputs(server: str, history_entry: dict, out_dir: Path) -> list[str]:
    downloaded: list[str] = []
    outputs = history_entry.get("outputs", {})
    for node_output in outputs.values():
        for values in node_output.values():
            if not isinstance(values, list):
                continue
            for item in values:
                if not isinstance(item, dict) or "filename" not in item:
                    continue
                filename = item["filename"]
                is_mesh = filename.lower().endswith(MESH_EXTENSIONS)
                is_image = filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                if not (is_mesh or is_image):
                    continue
                path = _download(
                    server,
                    filename,
                    item.get("subfolder", ""),
                    item.get("type", "output"),
                    out_dir,
                )
                downloaded.append(str(path))
    return downloaded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mode", required=True, choices=["pixal3d", "trellis2"])
    parser.add_argument("--resolution", required=True, type=int, choices=[1024, 1536])
    parser.add_argument("--out", required=True)
    parser.add_argument("--template", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    object_info = fetch_object_info(args.server)
    workflow = load_template(args.template)
    prompt = ui_workflow_to_api_prompt(workflow, object_info)

    filename = upload_image(args.server, Path(args.image))
    set_loadimage_filenames(prompt, filename)

    gating_booleans = apply_mode(prompt, workflow, args.mode)
    set_resolution(prompt, args.resolution)
    if args.seed is not None:
        set_seed(prompt, args.seed)

    print(f"mode={args.mode} gated by PrimitiveBoolean node(s) {gating_booleans}")

    client_id = str(uuid.uuid4())
    start = time.monotonic()
    prompt_id = submit_prompt(args.server, prompt, client_id)
    print(f"submitted prompt {prompt_id}")

    history_entry = poll_history(args.server, prompt_id, args.timeout)
    elapsed = time.monotonic() - start

    branch_problems = verify_3d_branch_executed(prompt, history_entry)

    downloaded = collect_outputs(args.server, history_entry, out_dir)
    mesh_paths = [p for p in downloaded if p.lower().endswith(MESH_EXTENSIONS)]
    if not mesh_paths:
        branch_problems.append(
            f"no mesh output (extensions {MESH_EXTENSIONS}) was produced; downloaded: {downloaded}"
        )

    run_summary = {
        "prompt_id": prompt_id,
        "mode": args.mode,
        "resolution": args.resolution,
        "elapsed_seconds": elapsed,
        "outputs": downloaded,
        "status_messages": history_entry.get("status", {}).get("messages", []),
        "ok": not branch_problems,
        "problems": branch_problems,
    }
    (out_dir / "run.json").write_text(json.dumps(run_summary, indent=2))

    if branch_problems:
        print("the 3D output branch did not run to completion:", file=sys.stderr)
        for problem in branch_problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    print(f"completed in {elapsed:.1f}s, {len(downloaded)} output file(s) downloaded to {out_dir}")
    print(f"GLB: {mesh_paths[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
