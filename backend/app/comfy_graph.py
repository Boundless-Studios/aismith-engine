"""Convert a ComfyUI UI-format workflow (nodes/links/widgets_values) into the
/prompt API format `{node_id: {"class_type": ..., "inputs": {...}}}`.

The UI format stores widget values positionally in each node's
`widgets_values`, in the order object_info lists that node's `input.required`
then `input.optional` entries, skipping any entry that is wired as a graph
socket instead. Whether an input is a socket or a widget is NOT determined by
its declared type (ComfyUI renders sockets for arbitrary custom types, and
widgets for arbitrary custom types too -- e.g. the V3 `COMBO` type and the
frontend-only `LOAD_3D` viewport-state type are both plain inline widgets);
it is determined by whether the node's own `inputs` list carries an entry of
that name at all. If it does, it is a socket: an unconnected optional socket
(`link` is null, e.g. an unwired `model_3d_info`) is skipped entirely (no
widgets_values slot, no output key); a connected one resolves through
`workflow["links"]` (following any `Reroute` chain), and if it was "converted
to input" from a widget (`"widget"` key present on the input entry) it still
reserves its widgets_values slot even though the value itself comes from the
link. If the name has no entry in `inputs` at all, it is a plain widget and
consumes the next `widgets_values` slot positionally.

Two widget shapes need extra handling:
  - An INT widget named `seed`/`noise_seed`, or carrying
    `control_after_generate` in its options, is followed by one extra
    `control_after_generate` string ("fixed"/"randomize"/...) that must be
    consumed and discarded.
  - `COMFY_DYNAMICCOMBO_V3` (e.g. RemeshMesh's `sign_mode`) stores its own
    selected option key, then that option's own `inputs.required`/`optional`
    sub-fields (looked up from `options["options"]` in object_info) inline,
    right after it, in their own declared order -- e.g. `sign_mode` picking
    "udf" is followed by "udf"'s `qef`, `drop_inverted_components`,
    `drop_enclosed_components` before the node's next real top-level input.
    The server expects those sub-fields as DOTTED keys, `"<combo_name>.
    <sub_name>"` (e.g. `sign_mode.qef`, confirmed against a live ComfyUI
    0.34.4 /prompt node_errors payload -- extra_info.input_name is literally
    "sign_mode.drop_inverted_components"), while the combo's own selection
    stays under its plain name (`sign_mode="udf"`).
    ComfyUI's server-side validator treats every one of the selected
    option's sub-fields as unconditionally required once that option is
    selected -- it does NOT tolerate an absent key the way it tolerates a
    missing ordinary top-level widget (which the executor defaults at run
    time). So if `widgets_values` runs out before reaching a sub-field (a
    real UI serialization this template exhibits when "advanced" widgets are
    left collapsed), that sub-field still must be emitted, falling back to
    its own declared `"default"` from object_info rather than being omitted.
"""

from __future__ import annotations

from typing import Any

SKIP_NODE_TYPES = frozenset({"Note", "MarkdownNote", "PrimitiveNode", "Reroute"})
DYNAMIC_COMBO_TYPE = "COMFY_DYNAMICCOMBO_V3"
_NO_VALUE = object()


def _ordered_input_specs(input_section: dict) -> list[tuple[str, list]]:
    ordered: list[tuple[str, list]] = []
    for section in ("required", "optional"):
        for name, spec in input_section.get(section, {}).items():
            ordered.append((name, spec))
    return ordered


def _has_control_after_generate(type_spec: Any, options: dict, name: str) -> bool:
    if type_spec != "INT":
        return False
    if "control_after_generate" in options:
        return True
    return name in ("seed", "noise_seed")


def _take_widget_value(widgets_values: Any, cursor: list[int], name: str) -> Any:
    if isinstance(widgets_values, dict):
        return widgets_values.get(name, _NO_VALUE)
    if isinstance(widgets_values, list):
        idx = cursor[0]
        if idx >= len(widgets_values):
            return _NO_VALUE
        cursor[0] += 1
        return widgets_values[idx]
    return _NO_VALUE


def _drop_widget_value(widgets_values: Any, cursor: list[int]) -> None:
    if isinstance(widgets_values, list) and cursor[0] < len(widgets_values):
        cursor[0] += 1


def _resolve_link_source(
    link: list, nodes_by_id: dict[int, dict], links_by_id: dict[int, list]
) -> list:
    from_node_id, from_slot = link[1], link[2]
    while True:
        source_node = nodes_by_id.get(from_node_id)
        if source_node is None or source_node.get("type") != "Reroute":
            return [str(from_node_id), from_slot]
        reroute_inputs = source_node.get("inputs") or []
        inner_link_id = reroute_inputs[0].get("link") if reroute_inputs else None
        inner_link = links_by_id.get(inner_link_id) if inner_link_id is not None else None
        if inner_link is None:
            return [str(from_node_id), from_slot]
        from_node_id, from_slot = inner_link[1], inner_link[2]


def _process_input(
    name: str,
    spec: list,
    *,
    sockets: dict[str, dict],
    widgets_values: Any,
    cursor: list[int],
    links_by_id: dict[int, list],
    nodes_by_id: dict[int, dict],
    api_inputs: dict[str, Any],
    require_value: bool = False,
    output_name: str | None = None,
) -> None:
    output_name = name if output_name is None else output_name
    type_spec = spec[0] if spec else None
    options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if options.get("hidden"):
        return

    socket = sockets.get(name)
    if socket is not None:
        link_id = socket.get("link")
        if link_id is not None:
            link = links_by_id.get(link_id)
            if link is not None:
                api_inputs[output_name] = _resolve_link_source(link, nodes_by_id, links_by_id)
            if "widget" in socket and widgets_values is not None:
                _drop_widget_value(widgets_values, cursor)
                if _has_control_after_generate(type_spec, options, name):
                    _drop_widget_value(widgets_values, cursor)
        return

    if type_spec == DYNAMIC_COMBO_TYPE:
        selected = _take_widget_value(widgets_values, cursor, name) if widgets_values is not None else _NO_VALUE
        if selected is _NO_VALUE:
            selected = options.get("default", _NO_VALUE)
        if selected is _NO_VALUE:
            return
        api_inputs[output_name] = selected
        option = next((o for o in options.get("options", []) if o.get("key") == selected), None)
        if option is None:
            return
        for sub_name, sub_spec in _ordered_input_specs(option.get("inputs", {})):
            _process_input(
                sub_name,
                sub_spec,
                sockets=sockets,
                widgets_values=widgets_values,
                cursor=cursor,
                links_by_id=links_by_id,
                nodes_by_id=nodes_by_id,
                api_inputs=api_inputs,
                require_value=True,
                output_name=f"{name}.{sub_name}",
            )
        return

    value = _take_widget_value(widgets_values, cursor, name) if widgets_values is not None else _NO_VALUE
    if value is _NO_VALUE and require_value:
        value = options.get("default", _NO_VALUE)
    if value is _NO_VALUE:
        return
    api_inputs[output_name] = value
    if _has_control_after_generate(type_spec, options, name):
        _drop_widget_value(widgets_values, cursor)


def ui_workflow_to_api_prompt(workflow: dict, object_info: dict) -> dict[str, dict]:
    nodes = workflow["nodes"]
    links = workflow.get("links", [])
    nodes_by_id = {node["id"]: node for node in nodes}
    links_by_id = {link[0]: link for link in links}

    prompt: dict[str, dict] = {}

    for node in nodes:
        node_type = node.get("type")
        if node_type in SKIP_NODE_TYPES:
            continue

        class_info = object_info.get(node_type)
        if class_info is None:
            raise ValueError(f"node {node['id']} has unknown class_type {node_type!r}")

        sockets = {inp["name"]: inp for inp in node.get("inputs", [])}
        widgets_values = node.get("widgets_values")
        cursor = [0]
        api_inputs: dict[str, Any] = {}

        for name, spec in _ordered_input_specs(class_info.get("input", {})):
            _process_input(
                name,
                spec,
                sockets=sockets,
                widgets_values=widgets_values,
                cursor=cursor,
                links_by_id=links_by_id,
                nodes_by_id=nodes_by_id,
                api_inputs=api_inputs,
            )

        for inp in node.get("inputs", []):
            name = inp["name"]
            if name in api_inputs or inp.get("link") is None:
                continue
            link = links_by_id.get(inp["link"])
            if link is not None:
                api_inputs[name] = _resolve_link_source(link, nodes_by_id, links_by_id)

        prompt[str(node["id"])] = {"class_type": node_type, "inputs": api_inputs}

    return prompt


def find_nodes(prompt: dict[str, dict], class_type: str) -> list[str]:
    return [node_id for node_id, node in prompt.items() if node.get("class_type") == class_type]
