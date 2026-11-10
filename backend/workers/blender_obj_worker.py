from __future__ import annotations
import json
import sys
from pathlib import Path
import bpy

def main():
    spec = json.loads(sys.argv[sys.argv.index("--") + 1])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    source, output = spec["input"], spec["output"]
    if Path(source).suffix.lower() in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=source)
    else:
        bpy.ops.wm.obj_import(filepath=source)
    bpy.ops.object.select_all(action="SELECT")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.obj_export(filepath=output, export_materials=True, export_normals=True, export_uv=True)

if __name__ == "__main__":
    main()
