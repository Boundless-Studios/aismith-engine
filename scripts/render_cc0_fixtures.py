"""Render the CC0-derived test fixtures and the T-pose guide.

Every image this script writes is a render of a CC0 3D model vendored with
Mesh2Motion (`vendor/mesh2motion-app`; the licence is `LICENSE-CC0.MD` there,
and `src/lib/RigModelVariations.ts` lists each model variation's licence),
posed and dressed here with primitive geometry and flat colours. No image
generator, no downloaded art and no earlier fixture is an input.

One command regenerates every image (paths resolve from this file, so the
working directory does not matter):

    blender --background --factory-startup \\
        --python scripts/render_cc0_fixtures.py

Options go after a literal `--`:

    --only NAME [NAME ...]   render only these outputs (PNG file stems)
    --out-dir DIR            write the PNGs flat into DIR instead of the repo

Determinism: the Workbench engine, fixed anti-aliasing, the Standard view
transform, no dithering and no metadata stamps, so a rerun on the same Blender
build and GPU reproduces the committed bytes. Another GPU may round
anti-aliased edge pixels differently; the PROVENANCE.md beside each set of
renders records the hashes the committed files have. Written against
Blender 5.1.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "vendor" / "mesh2motion-app" / "static" / "models-variation"
# All three are CC0 in both vendor/mesh2motion-app/LICENSE-CC0.MD and
# src/lib/RigModelVariations.ts. Do not add Sophia (CC-SA 4.0) or Jay,
# Sintel, Bunny or Carrot (CC-BY) here.
FEMALE_MANNEQUIN = MODELS / "human-female.glb"  # "Female Mannequin", Quaternius
MANNEQUIN = MODELS / "human-base.glb"  # "Mannequin", Quaternius
ZOMBIE = MODELS / "human-zombie.glb"  # "Zombie", Kenney.nl

FIXTURE_DIR = REPO / "backend" / "tests" / "fixtures" / "view_consistency"
POSE_GUIDE_DIR = REPO / "backend" / "app" / "assets" / "pose-guides"

FIXTURE_SIZE = 256
FIXTURE_BACKGROUND = (232, 232, 232)
FIXTURE_MARGIN = 1.10  # frame edge = 1.10x the figure's largest extent
POSE_GUIDE_SIZE = 1254
POSE_GUIDE_BACKGROUND = (255, 255, 255)
POSE_GUIDE_MARGIN = 1.08

# Flat outfit colours, sRGB.
JUMPSUIT = (104, 64, 150)
BOOTS = (64, 48, 40)
BELT = (40, 36, 34)
SKIN = (222, 184, 150)
ROBE = (92, 56, 132)
ROBE_TRIM = (150, 120, 60)
MANNEQUIN_GREY = (240, 240, 240)

# The floor-length robe: an elliptical bell that hugs the waist, clears the
# hips and flares to the floor. Profile points are (height, side-to-side
# radius, centre y) in metres, sized to the Female Mannequin's rest mesh;
# radii are interpolated smoothly between them. The front-to-back radius is
# 0.9x the side-to-side one, so the robe's profile view is no wider than its
# front view (a profile of the same robe must not read as widened).
ROBE_PROFILE = (
    (1.14, 0.135, -0.03),  # waist
    (0.95, 0.205, 0.01),  # hips
    (0.004, 0.30, 0.01),  # hem, just above the floor
)
ROBE_DEPTH_RATIO = 0.90
ROBE_SASH_HEIGHT = 0.06  # trim band below the waist edge
ROBE_HEM_TRIM_HEIGHT = 0.05  # trim band above the hem


@dataclass(frozen=True)
class Render:
    name: str
    directory: Path
    model: Path
    outfit: str  # "jumpsuit", "robe", "texture" or "grey"
    view: str  # "front", "left" or "back"
    leg_spread_degrees: float
    size: int
    background: tuple[int, int, int]
    margin: float
    camera: str = "ortho"  # or "perspective"
    level_thumbs: bool = False
    subdivision_levels: int = 0


RENDERS = (
    # Two clearly separated legs in a fitted jumpsuit and boots.
    Render("front-legs", FIXTURE_DIR, FEMALE_MANNEQUIN, "jumpsuit", "front", 4.0,
           FIXTURE_SIZE, FIXTURE_BACKGROUND, FIXTURE_MARGIN),
    # The same figure in a floor-length robe: one fused lower body.
    Render("front-robe-source", FIXTURE_DIR, FEMALE_MANNEQUIN, "robe", "front", 0.0,
           FIXTURE_SIZE, FIXTURE_BACKGROUND, FIXTURE_MARGIN),
    Render("left-robe-drift", FIXTURE_DIR, FEMALE_MANNEQUIN, "robe", "left", 0.0,
           FIXTURE_SIZE, FIXTURE_BACKGROUND, FIXTURE_MARGIN),
    Render("back-robe-drift", FIXTURE_DIR, FEMALE_MANNEQUIN, "robe", "back", 0.0,
           FIXTURE_SIZE, FIXTURE_BACKGROUND, FIXTURE_MARGIN),
    # A different two-legged character, rendered consistently in three views.
    Render("front-zombie", FIXTURE_DIR, ZOMBIE, "texture", "front", 0.0,
           FIXTURE_SIZE, FIXTURE_BACKGROUND, FIXTURE_MARGIN),
    Render("left-zombie-good", FIXTURE_DIR, ZOMBIE, "texture", "left", 0.0,
           FIXTURE_SIZE, FIXTURE_BACKGROUND, FIXTURE_MARGIN),
    Render("back-zombie-good", FIXTURE_DIR, ZOMBIE, "texture", "back", 0.0,
           FIXTURE_SIZE, FIXTURE_BACKGROUND, FIXTURE_MARGIN),
    # The image-edit pose reference: a plain grey mannequin, strict T-pose.
    Render("humanoid-t-pose-v2", POSE_GUIDE_DIR, MANNEQUIN, "grey", "front", 3.5,
           POSE_GUIDE_SIZE, POSE_GUIDE_BACKGROUND, POSE_GUIDE_MARGIN,
           camera="perspective", level_thumbs=True, subdivision_levels=1),
)


def srgb_to_linear(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    def channel(value: float) -> float:
        c = value / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return tuple(channel(value) for value in rgb)


def background_linear(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """World colour that lands exactly on `rgb` in the 8-bit output.

    Workbench's 32-sample accumulation loses a fraction of a count, so an
    exact value writes one count low (232 becomes 231, 255 becomes 254).
    Aiming at the middle of the 8-bit bucket lands on the intended value."""
    return srgb_to_linear(tuple(value + 0.5 for value in rgb))


def flat_material(name: str, rgb: tuple[int, int, int]) -> bpy.types.Material:
    """A material whose Workbench colour and Principled base colour agree."""
    linear = (*srgb_to_linear(rgb), 1.0)
    material = bpy.data.materials.new(name)
    material.diffuse_color = linear
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = linear
        principled.inputs["Roughness"].default_value = 0.8
    return material


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_character(path: Path) -> tuple[bpy.types.Object, list[bpy.types.Object]]:
    """Import a Mesh2Motion GLB; return its armature and skinned meshes.

    Each human variation also carries an unparented 42-vertex "Icosphere"
    helper mesh at the origin; it is not part of the character and is
    removed."""
    bpy.ops.import_scene.gltf(filepath=str(path))
    armatures = [ob for ob in bpy.data.objects if ob.type == "ARMATURE"]
    if len(armatures) != 1:
        raise RuntimeError(f"{path.name}: expected one armature, found {len(armatures)}")
    armature = armatures[0]
    meshes = []
    for ob in list(bpy.data.objects):
        if ob.type != "MESH":
            continue
        if ob.parent is armature:
            meshes.append(ob)
        else:
            bpy.data.objects.remove(ob)
    if not meshes:
        raise RuntimeError(f"{path.name}: no skinned mesh")
    if armature.matrix_world != Matrix.Identity(4):
        raise RuntimeError(f"{path.name}: armature has an object transform")
    return armature, meshes


def rotate_bone(armature: bpy.types.Object, name: str, axis: str, degrees: float) -> None:
    """Rotate a pose bone about a world axis through its current head."""
    bone = armature.pose.bones[name]
    pivot = bone.head.copy()
    rotation = Matrix.Rotation(math.radians(degrees), 4, axis)
    bone.matrix = Matrix.Translation(pivot) @ rotation @ Matrix.Translation(-pivot) @ bone.matrix
    bpy.context.view_layer.update()


def spread_legs(armature: bpy.types.Object, degrees: float) -> None:
    """Swing each straight leg outward at the hip by `degrees` and turn the
    foot back so the sole stays flat. The figure faces -Y; its left side
    is +X."""
    if not degrees:
        return
    rotate_bone(armature, "thigh_l", "Y", -degrees)
    rotate_bone(armature, "foot_l", "Y", degrees)
    rotate_bone(armature, "thigh_r", "Y", degrees)
    rotate_bone(armature, "foot_r", "Y", -degrees)


def level_thumbs(armature: bpy.types.Object) -> None:
    """Swing each thumb up into the horizontal plane of the flat, palm-down
    hand. At rest the mannequin's thumbs angle down and forward, which reads
    from the front as a hooked hand rather than an open one."""
    for side in ("l", "r"):
        base = armature.pose.bones[f"thumb_01_{side}"]
        tip = armature.pose.bones[f"thumb_04_leaf_{side}"]
        direction = tip.tail - base.head
        level = Vector((direction.x, direction.y, 0.0))
        rotation = direction.rotation_difference(level).to_matrix().to_4x4()
        pivot = base.head.copy()
        base.matrix = Matrix.Translation(pivot) @ rotation @ Matrix.Translation(-pivot) @ base.matrix
        bpy.context.view_layer.update()


def dress_jumpsuit(mesh: bpy.types.Object) -> None:
    """Replace the mannequin's materials with a fitted jumpsuit: skin on the
    head, neck and hands, a belt at the waist, boots to mid-calf. Zones are
    picked by each face's rest-pose position (the mannequin is in a T-pose
    at rest, so the hands are the faces beyond the wrists)."""
    materials = {
        "jumpsuit": flat_material("Jumpsuit", JUMPSUIT),
        "skin": flat_material("Skin", SKIN),
        "boots": flat_material("Boots", BOOTS),
        "belt": flat_material("Belt", BELT),
    }
    mesh.data.materials.clear()
    order = list(materials)
    for key in order:
        mesh.data.materials.append(materials[key])
    world = mesh.matrix_world
    for polygon in mesh.data.polygons:
        centre = world @ polygon.center
        if centre.z > 1.53 or abs(centre.x) > 0.66:
            zone = "skin"
        elif centre.z < 0.30:
            zone = "boots"
        elif 0.97 < centre.z < 1.03:
            zone = "belt"
        else:
            zone = "jumpsuit"
        polygon.material_index = order.index(zone)


def robe_section(z: float) -> tuple[float, float]:
    """(side-to-side radius, centre y) of the robe at height `z`, eased
    between the ROBE_PROFILE points with a smoothstep."""
    for (z_high, r_high, y_high), (z_low, r_low, y_low) in zip(ROBE_PROFILE, ROBE_PROFILE[1:]):
        if z_low <= z <= z_high:
            t = (z_high - z) / (z_high - z_low)
            t = t * t * (3.0 - 2.0 * t)
            return r_high + (r_low - r_high) * t, y_high + (y_low - y_high) * t
    raise ValueError(z)


def add_robe() -> bpy.types.Object:
    """A floor-length robe: an elliptical bell (64 segments, 40 rings) from
    the waist to the floor, with a sash band at the waist and a trim band at
    the hem. Open at both ends; neither opening faces a horizontal camera."""
    mesh = bpy.data.meshes.new("Robe")
    robe = bpy.data.objects.new("Robe", mesh)
    bpy.context.scene.collection.objects.link(robe)
    robe.data.materials.append(flat_material("Robe", ROBE))
    robe.data.materials.append(flat_material("RobeTrim", ROBE_TRIM))

    top_z, hem_z = ROBE_PROFILE[0][0], ROBE_PROFILE[-1][0]
    segments, rings = 64, 40
    bm = bmesh.new()
    grid = []
    for ring in range(rings + 1):
        z = top_z + (hem_z - top_z) * ring / rings
        radius, centre_y = robe_section(z)
        row = []
        for segment in range(segments):
            angle = 2.0 * math.pi * segment / segments
            row.append(bm.verts.new((
                radius * math.cos(angle),
                centre_y + radius * ROBE_DEPTH_RATIO * math.sin(angle),
                z,
            )))
        grid.append(row)
    for ring in range(rings):
        for segment in range(segments):
            following = (segment + 1) % segments
            face = bm.faces.new((
                grid[ring][segment], grid[ring + 1][segment],
                grid[ring + 1][following], grid[ring][following],
            ))
            mid_z = (grid[ring][segment].co.z + grid[ring + 1][segment].co.z) / 2.0
            trim = mid_z > top_z - ROBE_SASH_HEIGHT or mid_z < hem_z + ROBE_HEM_TRIM_HEIGHT
            face.material_index = 1 if trim else 0
            face.smooth = True
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    return robe


def dress_grey(mesh: bpy.types.Object) -> None:
    grey = flat_material("Mannequin", MANNEQUIN_GREY)
    mesh.data.materials.clear()
    mesh.data.materials.append(grey)
    for polygon in mesh.data.polygons:
        polygon.material_index = 0


def evaluated_bounds() -> tuple[Vector, Vector]:
    """World-space bounds of every mesh vertex after posing."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    low = Vector((math.inf,) * 3)
    high = Vector((-math.inf,) * 3)
    for ob in bpy.context.scene.objects:
        if ob.type != "MESH":
            continue
        evaluated = ob.evaluated_get(depsgraph)
        data = evaluated.to_mesh()
        for vertex in data.vertices:
            point = evaluated.matrix_world @ vertex.co
            low = Vector(map(min, low, point))
            high = Vector(map(max, high, point))
        evaluated.to_mesh_clear()
    return low, high


VIEW_ROTATIONS = {
    # Euler XYZ for a camera looking horizontally at a figure facing -Y.
    "front": (math.pi / 2, 0.0, 0.0),  # from -Y
    "left": (math.pi / 2, 0.0, math.pi / 2),  # from +X: the figure's left side
    "back": (math.pi / 2, 0.0, math.pi),  # from +Y
}


def place_camera(spec: Render) -> None:
    """Frame the posed figure. The frame side is the figure's largest
    extent on any axis times the margin, so every view of one character is
    rendered at the same scale."""
    scene = bpy.context.scene
    low, high = evaluated_bounds()
    centre = (low + high) / 2.0
    frame = max(high - low) * spec.margin
    data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.rotation_euler = VIEW_ROTATIONS[spec.view]
    distance = 10.0
    if spec.camera == "perspective":
        # A long lens from shoulder height with a vertical lens shift: the
        # horizontal arms stay horizontal in the frame, verticals stay
        # vertical, and the tops of the feet are visible.
        data.type = "PERSP"
        data.lens = 85.0
        data.sensor_fit = "AUTO"
        data.sensor_width = 36.0
        half_fov = math.atan(data.sensor_width / 2.0 / data.lens)
        distance = frame / 2.0 / math.tan(half_fov)
        eye_z = 1.44
        data.shift_y = (centre.z - eye_z) / frame
    else:
        data.type = "ORTHO"
        data.ortho_scale = frame
        eye_z = centre.z
    data.clip_start = 0.1
    data.clip_end = 100.0
    if spec.view == "front":
        camera.location = (centre.x, low.y - distance, eye_z)
    elif spec.view == "back":
        camera.location = (centre.x, high.y + distance, eye_z)
    else:
        camera.location = (high.x + distance, centre.y, eye_z)


def configure_render(spec: Render, color_type: str) -> None:
    scene = bpy.context.scene
    render = scene.render
    render.engine = "BLENDER_WORKBENCH"
    render.resolution_x = spec.size
    render.resolution_y = spec.size
    render.resolution_percentage = 100
    render.pixel_aspect_x = 1.0
    render.pixel_aspect_y = 1.0
    render.film_transparent = False
    render.dither_intensity = 0.0
    for attribute in dir(render):
        if attribute.startswith("use_stamp"):
            setattr(render, attribute, False)
    scene.display.render_aa = "32"
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.studio_light = "Default"
    shading.use_world_space_lighting = False
    shading.color_type = color_type
    shading.show_specular_highlight = False
    shading.show_shadows = False
    shading.show_cavity = False
    shading.show_object_outline = False
    shading.show_xray = False
    world = bpy.data.worlds.new("Background")
    world.color = background_linear(spec.background)
    scene.world = world
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    image = render.image_settings
    image.file_format = "PNG"
    image.color_mode = "RGB"
    image.color_depth = "8"
    image.compression = 15


def render_one(spec: Render, out_dir: Path | None) -> Path:
    reset_scene()
    armature, meshes = import_character(spec.model)
    spread_legs(armature, spec.leg_spread_degrees)
    if spec.level_thumbs:
        level_thumbs(armature)
    if spec.subdivision_levels:
        for mesh in meshes:
            modifier = mesh.modifiers.new("Subdivision", "SUBSURF")
            modifier.levels = spec.subdivision_levels
            modifier.render_levels = spec.subdivision_levels
    color_type = "MATERIAL"
    if spec.outfit in ("jumpsuit", "robe"):
        for mesh in meshes:
            dress_jumpsuit(mesh)
        if spec.outfit == "robe":
            add_robe()
    elif spec.outfit == "grey":
        for mesh in meshes:
            dress_grey(mesh)
    elif spec.outfit == "texture":
        color_type = "TEXTURE"  # the model's own embedded texture
    else:
        raise ValueError(spec.outfit)
    bpy.context.view_layer.update()
    place_camera(spec)
    configure_render(spec, color_type)
    target = (out_dir or spec.directory) / f"{spec.name}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(target)
    bpy.ops.render.render(write_still=True)
    return target


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="render_cc0_fixtures.py")
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    names = {spec.name for spec in RENDERS}
    unknown = set(args.only or ()) - names
    if unknown:
        parser.error(f"unknown render(s): {', '.join(sorted(unknown))}")
    print(f"Blender {bpy.app.version_string}")
    for spec in RENDERS:
        if args.only and spec.name not in args.only:
            continue
        target = render_one(spec, args.out_dir)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        print(f"RENDERED {digest}  {target}")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
