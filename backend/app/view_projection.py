"""S2 view-projection texturing: bake the T-pose conditioning views onto the
canonical mesh's own UV atlas (pure numpy -- no bpy import).

TRELLIS is conditioned on four synthesized T-pose views (front/back/left/
right PNGs, see `trellis2_fp8_worker.py:_prepare_multiview_foregrounds`), but
its own texture head only ever LOOKS at those views; it does not literally
copy their pixels onto the mesh it generates, so a generated character can
end up plausible-looking but not conditioning-accurate (a shirt one shade
off, a logo TRELLIS invented instead of reproduced). Those same four PNGs are
already lying around as artifacts, are already known-aligned to the T-pose
mesh (that is what "conditioning" means), and are strictly higher-resolution
evidence than anything the texture head painted. Projecting them back onto
the canonical UV atlas -- one blended orthographic re-texture per matched
camera -- recovers that evidence deterministically instead of asking a
generative model to reproduce it a second time.

Method, end to end: for every conditioning role, build the SAME zero-tilt
orthographic camera basis the Blender worker renders with
(`ortho_view_basis`), analytically project the atlas's own rasterized
world-space texels through it (`project_points`), and score each texel's
visibility to that camera by depth agreement with a rendered depth map and
membership in the PNG's own keyed subject silhouette
(`view_texel_weights`). The four views' contributions are blended by that
score (`blend_views`), falling back to the mesh's EXISTING baked texture
wherever no view contributes enough weight -- so this can only add
conditioning-accurate detail, never regress coverage below what TRELLIS
already produced.

Coordinate conventions, load-bearing for every function below. Getting any
one of these backwards is exactly the failure mode this module exists to
prevent, and it fails SILENTLY: every lookup still returns a number, just
the wrong one, which reads as "nothing occludes/faces the camera" rather
than as the plumbing bug it is (2026-09-04 live GPU-box run: every role's
`per_role_coverage` came back exactly 0.0, not because nothing matched but
because two of these conventions were inconsistent -- see git history on
this docstring for the postmortem).

* World frame is Blender's Z-up house frame; the canonical mesh's front
  faces -Y (see `blender_worker.py`'s `render_canonicalize_similarity_views`
  docstring). `ROLE_AZIMUTH_DEGREES` is measured the same way that camera's
  azimuth is.
* `direction` (a `ViewBasis` field) is the camera's OFFSET from the subject
  center -- the camera sits at `center + direction * distance` for some
  distance, not the direction the camera looks. `forward` is `-direction`,
  the camera's actual view axis (from camera into the scene).
* Facing sign: a texel's weight rewards `dot(normal, toward_camera)`, where
  `toward_camera` is the unit vector from the texel toward the camera. For
  an ORTHOGRAPHIC camera every view ray is parallel, so `toward_camera` is
  the same for every point in the scene and equals `basis.direction` exactly
  (not an approximation) -- `direction` points from `center` to the camera,
  which is the same direction from ANY point to the camera when the camera
  is infinitely far along it. Concretely: for the "front" role (camera at
  -Y looking toward +Y), a texel on the mesh's -Y-facing surface -- the one
  actually visible to that camera -- has a normal near `(0, -1, 0)`, which
  is `basis.direction` for that role, so `dot(normal, basis.direction)` is
  positive there. Using `basis.forward` instead would reward the mesh's FAR,
  camera-facing-away side.
* Horizontal handedness: `right = cross(forward, up)`, matching Blender's
  camera X axis. For the front role (camera at -Y looking +Y) world +X maps
  to a LARGER column index; for the left role (camera at +X looking -X)
  world +Y does. `test_projection_conformance` renders the SAME cameras in
  Blender and asserts this numerically; the reversed cross product passes
  every numpy-only test and mirrors every view.
* Pixel rows are BOTTOM-UP: row 0 is v = 0 / the bottom of the image, row
  H-1 is v = 1 / the top -- Blender's own `image.pixels` buffer convention
  (UV origin bottom-left), not the top-down convention common in general
  image processing. `project_points` maps world +Z to INCREASING row index
  (moving up the image increases the row number) and `rasterize_atlas` maps
  a UV's v directly to row (`v * H`, not `(1 - v) * H`). This is deliberate,
  not incidental: every buffer this module's arrays are ever compared
  against or written into -- a PNG loaded via Blender's image loader, an EXR
  depth pass loaded the same way, the atlas texture's own prior bake -- is
  ALSO bottom-up (Blender loads a conventionally top-down PNG file into a
  bottom-up buffer), so matching that convention here means the Blender
  worker needs zero `np.flipud` calls anywhere to keep them aligned. Do not
  "fix" this to read more intuitively without flipping every one of those
  buffers to match.
* Depth reference point: `project_points`'s `depth` is measured from
  `center` (`dot(point - center, forward)`), NOT from the camera's actual
  position -- an intentionally different reference frame from Blender's own
  Z/Depth render pass, which is the ABSOLUTE distance from the camera to the
  point along its view axis. The two differ by the constant `camera_distance`
  the camera was placed at (`camera_location = center + direction *
  camera_distance`): `blender_z = depth_from_center + camera_distance`.
  `view_texel_weights` takes `camera_distance` explicitly and does this
  conversion itself so a caller handing it a raw Blender Z-pass gets a
  meaningful `depth_bias` comparison rather than one that is off by
  `camera_distance` (typically many times larger than any reasonable bias)
  and therefore rejects nearly everything.

Perspective cameras (the source-pose chain). A synthesized conditioning view
can be rendered from whatever camera we like, so the four T-pose roles are
always zero-tilt orthographic; a PHOTO cannot -- it is a perspective view
from an unknown camera, and projecting its pixels back over a mesh rebuilt
from it needs that camera reproduced, not approximated by an ortho stand-in.
`PerspectiveCamera` is that path: `source_fidelity.SourceCamera`'s own
yaw/pitch/fov_x/center/distance, a pinhole model matching Blender's
(`sensor_fit = HORIZONTAL`, principal point at the image centre, square
pixels), built by `tilted_view_basis` so the basis matches Blender's
`to_track_quat("-Z", "Y")` placement at any pitch. Both kinds share the
`depth`-from-`center` frame above -- for a perspective camera Blender's Z
pass is likewise the distance along the view axis, so `blender_z =
depth_from_center + camera_distance` holds unchanged -- and both are passed
around as one `ProjectionCamera` so nothing downstream assumes orthographic.
Two things DO differ per kind and live on the camera: `facing`
(`dot(normal, toward_camera)`, where `toward_camera` is a constant only for
an orthographic camera) and `world_units_per_pixel` (a depth-bias scale).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .texture_bake import PALETTE_BACKGROUND_KEY_TOLERANCE, linear_rgb_to_oklab, srgb_to_linear

# Blender Z-up house frame; azimuth measured around the world Z axis with the
# canonical mesh's front facing -Y (blender_worker.py's own convention).
ROLE_AZIMUTH_DEGREES: dict[str, float] = {
    "front": 0.0,
    "back": 180.0,
    "left": 90.0,
    "right": -90.0,
}

# 3/4 guide-view roles (task9, mesh-guided view synthesis): halfway between
# two of the four base roles, same azimuth convention/sign as ROLE_AZIMUTH_DEGREES.
EXTRA_ROLE_AZIMUTH_DEGREES: dict[str, float] = {
    "front-left": 45.0,
    "front-right": -45.0,
    "back-left": 135.0,
    "back-right": -135.0,
}

WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# Route-B projection-quality defaults (bead s2-view-projection, 2026-09-05
# GPU-box finding: equal-weight view blending washes the face, grazing texels
# smear coats, and 70% of the atlas falls back to the blurred bake). All three
# are settings on `project_views`; documented together here since that is the
# one place a caller needs to reconcile numeric defaults against the report.
DEFAULT_ROLE_PRIORITY: dict[str, float] = {"front": 1.0, "back": 0.7, "left": 0.5, "right": 0.5}
# cos(70 deg): below this a texel's projected sample is foreshortened past
# use, not merely dim -- the source of the "vertical smear streaks" finding.
DEFAULT_MIN_FACING = 0.35
DEFAULT_FILL_RADIUS = 12


@dataclass(frozen=True)
class ViewBasis:
    """One zero-tilt orthographic camera's basis, in world space.

    `direction` is the camera's offset from the subject center (the camera
    sits at `center + direction * distance`); `forward` (`-direction`) is
    its view axis. `right` and `up` span the camera's image plane, with
    `up` always world +Z (a zero-tilt camera never rolls, so world Z always
    maps straight to image up regardless of azimuth).
    """

    role: str
    direction: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray


def azimuth_view_basis(role: str, azimuth_degrees: float) -> ViewBasis:
    """The zero-tilt orthographic camera basis for an ARBITRARY azimuth
    (task9: 3/4 guide-view roles and explicit per-role cameras) -- same maths
    as `ortho_view_basis`, generalized off the fixed four-role table.

    Deliberately zero-tilt (unlike the appearance-similarity gate's camera,
    which tilts +0.25 -- see blender_worker.py ~1229): a projection camera
    must match the T-pose conditioning renders exactly, which were rendered
    with no tilt at all.
    """
    angle = math.radians(azimuth_degrees)
    direction = np.array([math.sin(angle), -math.cos(angle), 0.0], dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    forward = -direction
    # forward x up, NOT up x forward: Blender's camera X axis is Y_cam x Z_cam
    # with Z_cam = -forward, i.e. +X world for the front camera. The reversed
    # cross product mirrors every view horizontally -- caught by the
    # projection conformance fixture on 2026-09-05 (side views 0.67 -> 0.99
    # within-bias once un-mirrored), invisible to palette metrics on a
    # near-symmetric T-pose.
    right = np.cross(forward, WORLD_UP)
    right = right / np.linalg.norm(right)
    return ViewBasis(role=role, direction=direction, right=right, up=WORLD_UP.copy(), forward=forward)


def tilted_view_basis(role: str, yaw_degrees: float, pitch_degrees: float) -> ViewBasis:
    """The camera basis for an arbitrary yaw AND pitch -- what a PHOTO's own
    camera needs, since nothing constrains it to the zero tilt a synthesized
    conditioning view is rendered with.

    `direction` is `source_fidelity.SourceCamera.direction()` exactly, same
    convention (front faces -Y, +Z up, yaw about +Z, pitch > 0 looks DOWN at
    the subject). `right` stays horizontal -- `cross(forward, WORLD_UP)`,
    Blender's camera X axis, as in `azimuth_view_basis` -- and `up` becomes
    `cross(right, forward)`, the camera's true image-up, which equals world
    +Z only at zero pitch. That is precisely the frame Blender's own
    `to_track_quat("-Z", "Y")` placement produces (`render_source_view`), so
    a render taken there and this projection agree at any pitch.

    Reduces to `azimuth_view_basis(role, yaw_degrees)` at pitch 0.
    """
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    direction = np.array(
        [math.sin(yaw) * math.cos(pitch), -math.cos(yaw) * math.cos(pitch), math.sin(pitch)],
        dtype=np.float64,
    )
    direction = direction / np.linalg.norm(direction)
    forward = -direction
    right = np.cross(forward, WORLD_UP)
    right_length = float(np.linalg.norm(right))
    if right_length < 1e-9:
        raise ValueError(
            f"tilted_view_basis: pitch {pitch_degrees} looks straight down the world up axis, "
            "which leaves the camera's roll undefined"
        )
    right = right / right_length
    return ViewBasis(
        role=role, direction=direction, right=right, up=np.cross(right, forward), forward=forward,
    )


def ortho_view_basis(role: str) -> ViewBasis:
    """The zero-tilt orthographic camera basis for one of the four table
    roles -- a thin wrapper over `azimuth_view_basis`/`ROLE_AZIMUTH_DEGREES`."""
    if role not in ROLE_AZIMUTH_DEGREES:
        raise ValueError(
            f"Unknown view role: {role!r} (expected one of {sorted(ROLE_AZIMUTH_DEGREES)})"
        )
    return azimuth_view_basis(role, ROLE_AZIMUTH_DEGREES[role])


def resolve_role_azimuth(role: str, cameras: dict | None = None) -> float:
    """The azimuth (degrees) to use for `role`: an explicit camera's own
    `azimuth_degrees` wins when `cameras` supplies one for this role (task9:
    a guide render's exact camera, not the fixed table), else the four-role
    table, else `EXTRA_ROLE_AZIMUTH_DEGREES` (3/4 roles with no explicit
    camera). Raises `ValueError` for a role none of those know about.
    """
    if cameras and role in cameras:
        return float(cameras[role]["azimuth_degrees"])
    if role in ROLE_AZIMUTH_DEGREES:
        return ROLE_AZIMUTH_DEGREES[role]
    if role in EXTRA_ROLE_AZIMUTH_DEGREES:
        return EXTRA_ROLE_AZIMUTH_DEGREES[role]
    raise ValueError(
        f"Unknown view role: {role!r} with no explicit camera "
        f"(expected one of {sorted({*ROLE_AZIMUTH_DEGREES, *EXTRA_ROLE_AZIMUTH_DEGREES})})"
    )


def project_points(
    points_xyz: np.ndarray,
    basis: ViewBasis,
    center: np.ndarray,
    ortho_scale: float,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points -> (px, py, depth) for a square ortho image of `image_size`
    pixels covering `ortho_scale` world units, centered on `center`.

    +Z maps to increasing row index (row 0 = v = 0 = the BOTTOM of the
    image, matching Blender's own bottom-up `image.pixels` convention -- see
    this module's docstring). `depth` increases moving AWAY from the camera
    along its view axis, measured from `center`, NOT from the camera's own
    position (also documented at the top of this module) -- a caller
    comparing this against a raw Blender Z/Depth pass must first subtract
    that camera's distance from `center`.
    """
    return _project_orthographic(points_xyz, basis, center, ortho_scale, image_size, image_size)


def _project_orthographic(
    points_xyz: np.ndarray,
    basis: ViewBasis,
    center: np.ndarray,
    ortho_scale: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`project_points` generalized to a non-square image: `ortho_scale` is
    world units across the image WIDTH (Blender's own `ortho_scale` under
    `sensor_fit = HORIZONTAL`) and the same units-per-pixel applies
    vertically, so a square image reproduces `project_points` exactly."""
    points = np.asarray(points_xyz, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    relative = points - center
    u = relative @ basis.right
    v = relative @ basis.up
    depth = relative @ basis.forward
    scale = float(width) / float(ortho_scale)
    px = float(width) / 2.0 + u * scale
    py = float(height) / 2.0 + v * scale
    return px, py, depth


# A point at or behind the camera plane has no image-plane position at all;
# the perspective projection reports it here instead -- finite (an `inf`
# casts to a garbage int64 in the bounds tests downstream) and far enough
# out of frame that every bounds test rejects it.
BEHIND_CAMERA_PIXEL = -1e9


def project_points_perspective(
    points_xyz: np.ndarray,
    basis: ViewBasis,
    center: np.ndarray,
    camera_distance: float,
    fov_x_degrees: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points -> (px, py, depth) for a PINHOLE camera sitting at
    `center + basis.direction * camera_distance` and looking at `center`,
    with horizontal field of view `fov_x_degrees` over a `width` x `height`
    image -- Blender's own camera model with `sensor_fit = HORIZONTAL` and
    the principal point at the image centre, i.e. exactly the camera
    `blender_worker.render_source_view` builds from a `SourceCamera`.

    Same return signature and the same two conventions as `project_points`:
    pixel rows are BOTTOM-UP (image-up maps to an INCREASING row index), and
    `depth` is measured from `center` along the view axis, NOT from the
    camera -- Blender's Z pass for this camera is `depth + camera_distance`,
    exactly as for the orthographic path, so both kinds share one depth
    reference frame downstream.

    Pixels are square, so the vertical focal length equals the horizontal one
    and the vertical field of view follows from `height / width`. A point at
    or behind the camera plane is reported at `BEHIND_CAMERA_PIXEL` with its
    true (non-positive) depth.
    """
    points = np.asarray(points_xyz, dtype=np.float64)
    location = np.asarray(center, dtype=np.float64) + basis.direction * float(camera_distance)
    relative = points - location
    x = relative @ basis.right
    y = relative @ basis.up
    z = relative @ basis.forward
    focal_px = (float(width) / 2.0) / math.tan(math.radians(float(fov_x_degrees)) / 2.0)
    in_front = z > 0.0
    safe_z = np.where(in_front, z, 1.0)
    px = np.where(in_front, float(width) / 2.0 + focal_px * x / safe_z, BEHIND_CAMERA_PIXEL)
    py = np.where(in_front, float(height) / 2.0 + focal_px * y / safe_z, BEHIND_CAMERA_PIXEL)
    return px, py, z - float(camera_distance)


@dataclass(frozen=True)
class OrthographicCamera:
    """One zero-tilt orthographic projection camera: the four T-pose
    conditioning roles and every explicit guide camera (task9).

    `ortho_scale` is world units across the image WIDTH; `camera_distance` is
    how far along `basis.direction` from `center` the camera was placed, the
    constant that converts a raw Blender Z pass into this module's
    depth-from-`center` frame.
    """

    basis: ViewBasis
    center: np.ndarray
    ortho_scale: float
    camera_distance: float
    width: int
    height: int

    def project(self, points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _project_orthographic(
            points_xyz, self.basis, self.center, self.ortho_scale, self.width, self.height,
        )

    def facing(self, normals: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
        """`dot(normal, toward_camera)`. Every ray is parallel here, so
        `toward_camera` is `basis.direction` for every point exactly (not an
        approximation) -- see this module's docstring; `points_xyz` is unused
        and taken only to match `PerspectiveCamera.facing`."""
        return np.asarray(normals, dtype=np.float64) @ self.basis.direction

    def world_units_per_pixel(self, depth_offset: float = 0.0) -> float:
        # `depth_offset` is accepted for signature parity with PerspectiveCamera; an
        # orthographic pixel covers the same world width at every depth.
        return float(self.ortho_scale) / float(self.width)


@dataclass(frozen=True)
class PerspectiveCamera:
    """One perspective projection camera -- a photo's own camera, described
    exactly as `source_fidelity.SourceCamera` describes it (`basis` from
    `tilted_view_basis(role, yaw, pitch)`, `camera_distance` its `distance`,
    `fov_x_degrees` its horizontal field of view, `width`/`height` the
    photo's own pixel size, which need not be square)."""

    basis: ViewBasis
    center: np.ndarray
    fov_x_degrees: float
    camera_distance: float
    width: int
    height: int

    def location(self) -> np.ndarray:
        return np.asarray(self.center, dtype=np.float64) + self.basis.direction * float(self.camera_distance)

    def project(self, points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return project_points_perspective(
            points_xyz, self.basis, self.center, self.camera_distance,
            self.fov_x_degrees, self.width, self.height,
        )

    def facing(self, normals: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
        """`dot(normal, toward_camera)` with `toward_camera` computed PER
        POINT: unlike the orthographic case the rays are not parallel, and at
        a photo's field of view the two differ by up to half of it -- enough
        to push a genuinely-visible texel under `min_facing` (cos 70 deg) at
        the edges of the frame."""
        toward = self.location() - np.asarray(points_xyz, dtype=np.float64)
        lengths = np.linalg.norm(toward, axis=-1, keepdims=True)
        toward = toward / np.clip(lengths, 1e-12, None)
        return np.einsum("ij,ij->i", np.asarray(normals, dtype=np.float64), toward)

    def world_units_per_pixel(self, depth_offset: float = 0.0) -> float:
        """At the plane `depth_offset` beyond the subject's centre (0 = the
        plane through `center`). A pixel covers less world at the near
        surface and more at the far one, so a depth bias sized for the whole
        subject takes the FAR plane (offset = half the mesh diagonal)."""
        half_width = (float(self.camera_distance) + float(depth_offset)) * math.tan(math.radians(float(self.fov_x_degrees)) / 2.0)
        return 2.0 * half_width / float(self.width)


# Either projection camera: everything from `_view_texel_masks` down takes
# one of these rather than assuming the orthographic geometry.
ProjectionCamera = OrthographicCamera | PerspectiveCamera


def fit_frame_to_subject(
    mesh_px_bbox: tuple[float, float, float, float],
    image_subject_bbox: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """A 2D similarity (uniform scale + translation) mapping the mesh's
    rendered pixel-space silhouette bbox onto the conditioning image's own
    subject bbox, both as (min_x, min_y, max_x, max_y).

    Uniform (not per-axis) scale so the mapping cannot distort the mesh's
    aspect ratio; the smaller of the two axis ratios is used so the mesh
    silhouette lands fully inside the image bbox rather than overflowing it.
    """
    mesh_min_x, mesh_min_y, mesh_max_x, mesh_max_y = mesh_px_bbox
    img_min_x, img_min_y, img_max_x, img_max_y = image_subject_bbox
    mesh_w = mesh_max_x - mesh_min_x
    mesh_h = mesh_max_y - mesh_min_y
    if mesh_w <= 0 or mesh_h <= 0:
        raise ValueError(f"fit_frame_to_subject needs a non-degenerate mesh bbox, got {mesh_px_bbox}")
    img_w = img_max_x - img_min_x
    img_h = img_max_y - img_min_y
    scale = min(img_w / mesh_w, img_h / mesh_h) if img_w > 0 and img_h > 0 else 1.0
    mesh_cx = (mesh_min_x + mesh_max_x) / 2.0
    mesh_cy = (mesh_min_y + mesh_max_y) / 2.0
    img_cx = (img_min_x + img_max_x) / 2.0
    img_cy = (img_min_y + img_max_y) / 2.0
    dx = img_cx - mesh_cx * scale
    dy = img_cy - mesh_cy * scale
    return float(scale), float(dx), float(dy)


def apply_frame(px: np.ndarray, py: np.ndarray, scale: float, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply a `fit_frame_to_subject` similarity to a batch of pixel coordinates."""
    return np.asarray(px, dtype=np.float64) * scale + dx, np.asarray(py, dtype=np.float64) * scale + dy


def invert_frame(scale: float, dx: float, dy: float) -> tuple[float, float, float]:
    """The inverse of a `fit_frame_to_subject` similarity: maps FROM the
    conditioning image's pixel space back to the render's own (unframed)
    camera pixel space. Used when the conditioning image's own subject
    cannot be keyed and the render's own silhouette has to stand in for it
    instead -- that silhouette lives in render space, so it has to be
    reprojected into the same space `subject_mask` is indexed in.
    """
    if scale == 0.0:
        raise ValueError("invert_frame needs a non-zero scale")
    inverse_scale = 1.0 / float(scale)
    return inverse_scale, -float(dx) * inverse_scale, -float(dy) * inverse_scale


def rasterize_atlas(
    uv_triangles: np.ndarray,
    world_triangles: np.ndarray,
    normal_triangles: np.ndarray,
    atlas_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill an `atlas_size`^2 atlas with barycentric-interpolated world
    position and normal, one UV triangle at a time.

    Each triangle's fill is vectorized over its own UV-space pixel bounding
    box (numpy meshgrid + barycentric weights), rather than a per-pixel
    Python loop; the loop over TRIANGLES stays in Python since Blender meshes
    rarely exceed a few hundred thousand of them and their per-triangle pixel
    footprints vary too much in size to batch cleanly.

    UV v maps directly to atlas row (row 0 = v = 0.0, the bottom of the
    texture) to match the bottom-up row order `image.pixels` uses, per this
    module's docstring -- NOT `(1 - v) * H`, which would flip it.
    """
    uv_triangles = np.asarray(uv_triangles, dtype=np.float64)
    world_triangles = np.asarray(world_triangles, dtype=np.float64)
    normal_triangles = np.asarray(normal_triangles, dtype=np.float64)
    if uv_triangles.ndim != 3 or uv_triangles.shape[1:] != (3, 2):
        raise ValueError(f"rasterize_atlas needs (n, 3, 2) uv_triangles, got {uv_triangles.shape}")
    if world_triangles.shape != (*uv_triangles.shape[:-1], 3):
        raise ValueError(
            f"rasterize_atlas needs world_triangles shaped {(*uv_triangles.shape[:-1], 3)}, got {world_triangles.shape}"
        )
    if normal_triangles.shape != world_triangles.shape:
        raise ValueError("rasterize_atlas needs normal_triangles to match world_triangles' shape")
    size = int(atlas_size)
    positions = np.zeros((size, size, 3), dtype=np.float64)
    normals = np.zeros((size, size, 3), dtype=np.float64)
    covered = np.zeros((size, size), dtype=bool)

    for tri_uv, tri_world, tri_normal in zip(uv_triangles, world_triangles, normal_triangles):
        px = tri_uv[:, 0] * size
        py = tri_uv[:, 1] * size
        min_x = max(0, int(np.floor(px.min())))
        max_x = min(size - 1, int(np.ceil(px.max())) - 1)
        min_y = max(0, int(np.floor(py.min())))
        max_y = min(size - 1, int(np.ceil(py.max())) - 1)
        if max_x < min_x or max_y < min_y:
            continue
        x0, y0 = px[0], py[0]
        x1, y1 = px[1], py[1]
        x2, y2 = px[2], py[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue
        cols = np.arange(min_x, max_x + 1)
        rows = np.arange(min_y, max_y + 1)
        col_grid, row_grid = np.meshgrid(cols, rows)
        sample_x = col_grid + 0.5
        sample_y = row_grid + 0.5
        w0 = ((y1 - y2) * (sample_x - x2) + (x2 - x1) * (sample_y - y2)) / denom
        w1 = ((y2 - y0) * (sample_x - x2) + (x0 - x2) * (sample_y - y2)) / denom
        w2 = 1.0 - w0 - w1
        eps = -1e-6
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        if not inside.any():
            continue
        sel_rows = row_grid[inside]
        sel_cols = col_grid[inside]
        weights = np.stack([w0[inside], w1[inside], w2[inside]], axis=-1)
        positions[sel_rows, sel_cols] = weights @ tri_world
        interpolated_normals = weights @ tri_normal
        norm_lengths = np.linalg.norm(interpolated_normals, axis=-1, keepdims=True)
        normals[sel_rows, sel_cols] = interpolated_normals / np.clip(norm_lengths, 1e-9, None)
        covered[sel_rows, sel_cols] = True

    return positions, normals, covered


def min_filter_neighborhood(array: np.ndarray, radius: int = 1) -> np.ndarray:
    """The minimum over a `(2*radius+1)`-square neighborhood of `array`,
    edges clamped (an out-of-bounds neighbor repeats the nearest in-bounds
    value, so an edge/corner is never pulled down by a phantom value).

    Used to make a depth comparison robust to a renderer's pixel filter
    blending the Z pass across a silhouette edge (a texel just inside a
    render's true edge can otherwise compare against a blended depth that
    belongs to neither surface) -- comparing against the neighborhood's
    minimum instead of the exact pixel is comparing against whichever
    nearby surface is genuinely closest to the camera, which is the correct
    occluder either way.
    """
    array = np.asarray(array, dtype=np.float64)
    if radius <= 0:
        return array
    padded = np.pad(array, radius, mode="edge")
    height, width = array.shape
    result = array.copy()
    for row_offset in range(-radius, radius + 1):
        for col_offset in range(-radius, radius + 1):
            window = padded[
                radius + row_offset : radius + row_offset + height,
                radius + col_offset : radius + col_offset + width,
            ]
            result = np.minimum(result, window)
    return result


def _erode_1d(mask: np.ndarray, axis: int, radius: int) -> np.ndarray:
    """AND a boolean mask with `radius` shifted copies of itself along one
    axis -- a pixel beyond the shifted array's edge counts as False (the
    image's border is background), which is what makes this also shrink a
    True region that touches the edge."""
    result = mask
    for offset in range(1, radius + 1):
        forward = np.zeros_like(mask)
        backward = np.zeros_like(mask)
        if axis == 0:
            forward[:-offset] = mask[offset:]
            backward[offset:] = mask[:-offset]
        else:
            forward[:, :-offset] = mask[:, offset:]
            backward[:, offset:] = mask[:, :-offset]
        result = result & forward & backward
    return result


def erode_mask(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Binary-erode a boolean mask by `radius` pixels: a `(2*radius+1)`-
    square structuring element, via a separable min-filter (a rectangular
    structuring element's erosion is exactly the AND of two 1D erosions,
    one per axis -- no scipy dependency needed).

    Shrinks the True region inward by `radius` pixels on every side, which
    is what keeps an imprecise subject-mask edge (a handful of background
    pixels the keying missed, or blur/anti-aliasing along the silhouette)
    from ever being sampled as "subject" -- the live GPU-box symptom this
    exists for was white background speckle along a coat and boot
    silhouette in the projected texture.

    Equivalent to `source_fidelity._erode_square` (orientation-agnostic,
    border counts as background) but kept separate: this module is imported
    inside Blender's bundled Python by the worker and must not grow a
    dependency on the fidelity gate's module.
    """
    eroded = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return eroded
    eroded = _erode_1d(eroded, axis=0, radius=radius)
    eroded = _erode_1d(eroded, axis=1, radius=radius)
    return eroded


HEAD_BAND_FRACTION = 0.20
HEAD_BAND_TOP_FRACTION = 0.40


def head_bbox(
    mask: np.ndarray,
    fraction: float = HEAD_BAND_FRACTION,
    top_fraction: float = HEAD_BAND_TOP_FRACTION,
    rows_bottom_up: bool = True,
) -> tuple[int, int, int, int] | None:
    """Head-only (x0, y0, x1, y1), x1/y1 exclusive, inside the top `fraction`
    of the subject's bbox. Same rule as the fidelity gate's head box: on a
    T-pose the outstretched arms also cross the band, but they enter it from
    BELOW while the head enters from ABOVE, so only columns that reach into
    the band's own top `top_fraction` count. None on an empty mask.

    `rows_bottom_up` (default True) follows this module's pixel-row
    convention (see the module docstring): row 0 is the BOTTOM of the image,
    so the head -- the top of the subject -- sits at the HIGHEST row indices
    and the band is measured downward from `ys.max()`. This is what the
    Blender worker's `_load_reference_pixels` buffer and its rendered-
    silhouette mask need; applying the top-down rule to them puts the "head
    band" on the boots. Pass `rows_bottom_up=False` for a conventional
    top-down mask (row 0 = top of the image).

    Deliberately NOT shared with `source_fidelity.head_bbox`: that copy runs
    on top-down PIL photo masks, so the two have opposite default row
    orientations even though the band rule is the same.
    """
    mask = np.asarray(mask, dtype=bool)
    if rows_bottom_up:
        # Flip to top-down, apply the top-down rule, then map the rows back:
        # top-down row r is bottom-up row (H - 1 - r), so an exclusive row
        # range [y0, y1) becomes [H - y1, H - y0).
        height = int(mask.shape[0])
        box = head_bbox(mask[::-1], fraction, top_fraction, rows_bottom_up=False)
        if box is None:
            return None
        x0, y0, x1, y1 = box
        return x0, height - y1, x1, height - y0
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    band_y1 = min(y1, y0 + max(1, round((y1 - y0) * fraction)))
    top_cut = y0 + max(1, round((band_y1 - y0) * top_fraction))
    qualifying = np.nonzero(mask[y0:top_cut, x0:x1].any(axis=0))[0]
    if qualifying.size == 0:
        return x0, y0, x1, band_y1
    col0, col1 = x0 + int(qualifying.min()), x0 + int(qualifying.max()) + 1
    band_ys, band_xs = np.nonzero(mask[y0:band_y1, col0:col1])
    if band_ys.size == 0:
        return x0, y0, x1, band_y1
    return col0 + int(band_xs.min()), y0 + int(band_ys.min()), col0 + int(band_xs.max()) + 1, y0 + int(band_ys.max()) + 1


def exclude_head(
    mask: np.ndarray, fraction: float = HEAD_BAND_FRACTION, rows_bottom_up: bool = True,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """The subject mask with its head box cleared, plus the box. A view drawn
    independently of the mesh (a T-pose front) sits on it at IoU ~0.92, not
    1:1, so projecting its face over one the texture already has in register
    double-exposes it -- excluding the head keeps the underlying texture there
    and still projects everything else the view sees.

    `rows_bottom_up` is forwarded to `head_bbox`; the default matches the
    bottom-up subject masks the Blender worker hands in (see `head_bbox`).
    """
    cleared = np.array(mask, dtype=bool, copy=True)
    box = head_bbox(cleared, fraction, rows_bottom_up=rows_bottom_up)
    if box is not None:
        x0, y0, x1, y1 = box
        cleared[y0:y1, x0:x1] = False
    return cleared, box


def mask_boundary_ring(mask: np.ndarray, radius: int = 4) -> np.ndarray:
    """The band of True pixels in `mask` within `radius` pixels of its own
    boundary (`mask & ~erode_mask(mask, radius)`) -- the region where
    genuine background bleed can still occur even after `erode_mask`, and
    the ONLY region a background-colour guard should ever look at. The deep
    interior (everything else `mask` covers) must never be touched by such
    a guard: a pale subject's skin can sit numerically close to a light
    background, and rejecting on colour alone anywhere inside the mask
    would reject real skin texels along with genuine bleed (2026-09-04
    GPU-box: a whole-mask colour guard flattened an alien's pale face).
    """
    mask = np.asarray(mask, dtype=bool)
    interior = erode_mask(mask, radius=radius)
    return mask & ~interior


def background_bleed_mask(
    colors: np.ndarray,
    subject_mask: np.ndarray,
    background_color: np.ndarray,
    sample_px: np.ndarray,
    sample_py: np.ndarray,
    ring_radius: int = 4,
    tolerance: float = PALETTE_BACKGROUND_KEY_TOLERANCE,
) -> np.ndarray:
    """Which sampled texel colours must be rejected as background bleed:
    within `tolerance` of `background_color` AND landing inside
    `mask_boundary_ring(subject_mask, ring_radius)` -- never true in
    `subject_mask`'s deep interior, however close a colour sits to the
    background, which is the entire point (see `mask_boundary_ring`'s
    docstring).

    `sample_px`/`sample_py` are the same conditioning-image pixel
    coordinates `colors` was bilinear-sampled at (`view_texel_weights`'s
    returned `sample_px`/`sample_py`); nearest-pixel lookup here, matching
    how `subject_mask` itself is looked up everywhere else in this module.
    """
    ring = mask_boundary_ring(subject_mask, radius=ring_radius)
    mask_height, mask_width = np.asarray(subject_mask).shape[:2]
    ring_ix = np.clip(np.round(np.asarray(sample_px, dtype=np.float64)).astype(np.int64), 0, mask_width - 1)
    ring_iy = np.clip(np.round(np.asarray(sample_py, dtype=np.float64)).astype(np.int64), 0, mask_height - 1)
    in_ring = ring[ring_iy, ring_ix]
    near_background = (
        np.abs(np.asarray(colors, dtype=np.float64) - np.asarray(background_color, dtype=np.float64)).sum(axis=-1)
        <= tolerance
    )
    return in_ring & near_background


def _view_texel_masks(
    positions: np.ndarray,
    normals: np.ndarray,
    camera: ProjectionCamera,
    depth_map: np.ndarray,
    depth_bias: float,
    subject_mask: np.ndarray,
    frame: tuple[float, float, float],
    depth_map_valid: np.ndarray | None = None,
    min_facing: float = 0.0,
) -> dict[str, Any]:
    """Shared per-texel visibility computation behind `view_texel_weights`
    and `view_texel_diagnostics` -- see `view_texel_weights`'s docstring for
    the conventions this implements. Returns flat (H*W,) boolean/float
    arrays plus `height`/`width`, so both public functions reshape and
    combine them however each needs to.

    `camera` is either projection kind (`OrthographicCamera` or
    `PerspectiveCamera`); nothing below assumes one or the other -- the
    projection, the facing dot product and the image size all come off the
    camera, and both kinds report `depth` in the same depth-from-`center`
    frame this module's docstring describes.

    `depth_map_valid` (render-space, same shape as `depth_map`) marks which
    pixels actually hit geometry -- see `view_texel_weights`'s docstring
    for why a background pixel must never be allowed to seed the
    neighborhood-minimum depth comparison.

    `min_facing` zeroes the weight of a texel whose `facing` falls below it
    even though `visible` is true: a grazing-angle sample is foreshortened
    past use (the coat "vertical smear streaks" finding), not merely dim.
    The default of 0.0 reproduces every prior weight exactly, since `facing`
    is already clipped to >= 0 below.
    """
    positions = np.asarray(positions, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if positions.shape != normals.shape or positions.shape[-1] != 3:
        raise ValueError("view_texel_weights needs matching (H, W, 3) positions and normals")
    height, width = positions.shape[:2]
    flat_positions = positions.reshape(-1, 3)
    flat_normals = normals.reshape(-1, 3)

    depth_map = np.asarray(depth_map, dtype=np.float64)
    # A depth map that does not match `image_size` is silently wrong rather
    # than loudly wrong: every lookup below still succeeds (clipped into
    # bounds) but samples the wrong texel's depth, which reads as ordinary
    # occlusion noise instead of the plumbing bug it actually is. Caught live
    # on the GPU box: Blender's compositor Viewer image does not track the
    # scene's render resolution in --background mode and kept returning a
    # stale 256x256 buffer against a 1024x1024 render.
    if depth_map.shape != (camera.height, camera.width):
        raise ValueError(
            f"view_texel_weights needs a depth_map shaped ({camera.height}, {camera.width}) "
            f"to match the camera's image size, got {depth_map.shape}"
        )
    # Convert Blender's Z/Depth pass (absolute distance from the CAMERA)
    # into this module's depth-from-`center` reference frame -- see this
    # module's docstring. Skipping this made every comparison below fail by
    # roughly `camera_distance`, which dwarfs any real `depth_bias` (the
    # 2026-09-04 GPU-box incident: occlusion rejected every single
    # candidate texel in every role).
    depth_map = depth_map - float(camera.camera_distance)
    if depth_map_valid is not None:
        depth_map_valid = np.asarray(depth_map_valid, dtype=bool)
        if depth_map_valid.shape != depth_map.shape:
            raise ValueError(
                f"view_texel_weights needs depth_map_valid shaped {depth_map.shape}, got {depth_map_valid.shape}"
            )
    # For the diagnostic range below, BEFORE any background pixels are
    # replaced with +inf (an `inf` in that range would defeat its purpose).
    depth_map_for_stats = depth_map if depth_map_valid is None else depth_map[depth_map_valid]
    if depth_map_valid is not None:
        # A background/no-hit pixel must never win a neighborhood minimum:
        # it holds whatever sentinel the renderer wrote there (observed on
        # the GPU box: near 0, which after the camera_distance shift above
        # reads as a huge NEGATIVE depth -- "closer than the camera itself"
        # -- and every real surface pixel within a THIN part of the mesh
        # (an arm, an ear) sits within one pixel of a background pixel
        # almost everywhere along its length, so the 3x3 min-filter dragged
        # that fake near value into nearly the whole part and failed its
        # depth test, while a thick part (a leg) mostly wasn't adjacent to
        # any background pixel and passed). +inf can never be a minimum.
        depth_map = np.where(depth_map_valid, depth_map, np.inf)
    # Compare against the 3x3 neighborhood MINIMUM, not the exact pixel: a
    # renderer's pixel filter blends the Z pass across a silhouette edge
    # (a low-sample Cycles render especially), so the single nearest pixel
    # can hold a depth that belongs to neither the near nor the far surface.
    # The minimum is always a genuine surface depth from somewhere nearby,
    # and is never further from the camera than the true nearest surface,
    # so it cannot manufacture a false "visible".
    depth_map_min = min_filter_neighborhood(depth_map, radius=1)

    render_px, render_py, depth = camera.project(flat_positions)
    render_ix = np.round(render_px).astype(np.int64)
    render_iy = np.round(render_py).astype(np.int64)
    in_render_bounds = (
        (render_ix >= 0) & (render_ix < camera.width) & (render_iy >= 0) & (render_iy < camera.height)
    )
    clipped_render_ix = np.clip(render_ix, 0, camera.width - 1)
    clipped_render_iy = np.clip(render_iy, 0, camera.height - 1)
    depth_at_render = depth_map_min[clipped_render_iy, clipped_render_ix]
    # One-sided, not `abs(diff) <= bias`: a texel in front of (at a smaller
    # depth than) the nearest rendered surface nearby is exactly what
    # "visible" means; only a texel BEHIND it is occluded.
    within_depth = depth <= depth_at_render + depth_bias

    scale, dx, dy = frame
    sample_px, sample_py = apply_frame(render_px, render_py, scale, dx, dy)
    sample_ix = np.round(sample_px).astype(np.int64)
    sample_iy = np.round(sample_py).astype(np.int64)
    mask_height, mask_width = np.asarray(subject_mask).shape[:2]
    in_sample_bounds = (
        (sample_ix >= 0) & (sample_ix < mask_width) & (sample_iy >= 0) & (sample_iy < mask_height)
    )
    clipped_sample_ix = np.clip(sample_ix, 0, mask_width - 1)
    clipped_sample_iy = np.clip(sample_iy, 0, mask_height - 1)
    subject = np.asarray(subject_mask, dtype=bool)[clipped_sample_iy, clipped_sample_ix]

    visible = in_render_bounds & within_depth & in_sample_bounds & subject
    facing = camera.facing(flat_normals, flat_positions)
    weights = np.where(visible & (facing >= min_facing), np.clip(facing, 0.0, None) ** 2, 0.0)
    return {
        "height": height,
        "width": width,
        "in_render_bounds": in_render_bounds,
        "within_depth": within_depth,
        "in_sample_bounds": in_sample_bounds,
        "subject": subject,
        "facing": facing,
        "weights": weights,
        "sample_px": sample_px,
        "sample_py": sample_py,
        # Raw per-texel depth (unmasked -- a caller restricts to `covered`
        # itself) and the whole render's depth range, both already shifted
        # into the center-relative reference frame: a caller can tell a
        # residual reference-frame offset from a genuine occlusion by eye,
        # since the two ranges should overlap substantially for a role that
        # is actually visible at all.
        "depth": depth,
        "depth_at_render": depth_at_render,
        "render_px": render_px,
        "render_py": render_py,
        "render_ix": render_ix,
        "render_iy": render_iy,
        "depth_map_min": float(depth_map_for_stats.min()) if depth_map_for_stats.size else None,
        "depth_map_max": float(depth_map_for_stats.max()) if depth_map_for_stats.size else None,
    }


def _orthographic_camera(
    basis: ViewBasis, center: np.ndarray, ortho_scale: float, image_size: int, camera_distance: float,
) -> OrthographicCamera:
    """The square orthographic camera the `basis`/`center`/`ortho_scale`/
    `image_size`/`camera_distance` argument form describes."""
    return OrthographicCamera(
        basis=basis, center=center, ortho_scale=ortho_scale,
        camera_distance=camera_distance, width=image_size, height=image_size,
    )


def view_texel_weights(
    positions: np.ndarray,
    normals: np.ndarray,
    basis: ViewBasis,
    center: np.ndarray,
    ortho_scale: float,
    image_size: int,
    depth_map: np.ndarray,
    depth_bias: float,
    subject_mask: np.ndarray,
    frame: tuple[float, float, float],
    camera_distance: float,
    depth_map_valid: np.ndarray | None = None,
    min_facing: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`view_texel_weights_for_camera` for a square orthographic camera given
    as its parts -- the four T-pose roles and every explicit guide camera."""
    return view_texel_weights_for_camera(
        positions, normals,
        _orthographic_camera(basis, center, ortho_scale, image_size, camera_distance),
        depth_map, depth_bias, subject_mask, frame,
        depth_map_valid=depth_map_valid, min_facing=min_facing,
    )


def view_texel_weights_for_camera(
    positions: np.ndarray,
    normals: np.ndarray,
    camera: ProjectionCamera,
    depth_map: np.ndarray,
    depth_bias: float,
    subject_mask: np.ndarray,
    frame: tuple[float, float, float],
    depth_map_valid: np.ndarray | None = None,
    min_facing: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-atlas-texel projection weight for one conditioning view.

    `depth_map` is Blender's raw Z/Depth render pass for the SAME camera
    `camera` describes -- either projection kind (`camera.camera_distance` is
    how far along `camera.basis.direction` from `camera.center` that camera
    was actually placed: `camera_location = center + basis.direction *
    camera_distance`). This function converts it into depth-from-`center`
    before comparing, so the caller must NOT pre-shift it itself. `frame`
    (from `fit_frame_to_subject`) maps the SAME projected coordinates into
    the CONDITIONING IMAGE's pixel space, which is where `subject_mask`
    lives and where the returned `sample_px`/`sample_py` are meant to be used
    (with `sample_bilinear`) to pull a colour off that image. The two pixel
    spaces usually nearly coincide -- the render camera is built from the
    same geometry the conditioning image was framed with -- but the frame
    absorbs whatever residual crop/scale mismatch remains.

    `depth_map_valid` (render-space, same shape as `depth_map`; typically the
    render's own alpha > 0.5) marks which pixels actually hit geometry. Pass
    it whenever `depth_map` can hold ANY background/no-hit pixels: without
    it, a background pixel's arbitrary sentinel value can win the 3x3
    neighborhood-minimum comparison below and falsely occlude real, nearby
    surface texels -- devastatingly so for a THIN part of the mesh (an arm,
    an ear), where almost every surface pixel sits within one pixel of a
    background pixel along its whole length (2026-09-05 GPU-box: exactly
    this starved the face and arms while a thick part, a leg, mostly
    survived because it mostly wasn't adjacent to any background pixel).

    A texel is visible when its analytically-projected depth agrees with the
    rendered depth map within `depth_bias` (survives occlusion) AND its
    frame-mapped sample lands inside `subject_mask` (is part of the
    conditioning image's keyed subject, not its background). Its weight is
    then how squarely its normal faces the camera, squared: see this
    module's docstring for why that is `dot(normal, basis.direction)` (and
    `camera.facing` for the perspective camera's per-point version of the
    same quantity), not `dot(normal, basis.forward)`.

    `min_facing` (default 0.0, reproducing every prior weight exactly) zeroes
    a texel's weight below that facing cutoff -- see `_view_texel_masks`.
    """
    masks = _view_texel_masks(
        positions, normals, camera, depth_map, depth_bias, subject_mask, frame,
        depth_map_valid=depth_map_valid, min_facing=min_facing,
    )
    height, width = masks["height"], masks["width"]
    return (
        masks["weights"].reshape(height, width),
        masks["sample_px"].reshape(height, width),
        masks["sample_py"].reshape(height, width),
    )


def view_texel_diagnostics(
    positions: np.ndarray,
    normals: np.ndarray,
    basis: ViewBasis,
    center: np.ndarray,
    ortho_scale: float,
    image_size: int,
    depth_map: np.ndarray,
    depth_bias: float,
    subject_mask: np.ndarray,
    frame: tuple[float, float, float],
    camera_distance: float,
    covered: np.ndarray,
    depth_map_valid: np.ndarray | None = None,
    min_facing: float = 0.0,
) -> dict[str, int]:
    """`view_texel_diagnostics_for_camera` for a square orthographic camera
    given as its parts."""
    return view_texel_diagnostics_for_camera(
        positions, normals,
        _orthographic_camera(basis, center, ortho_scale, image_size, camera_distance),
        depth_map, depth_bias, subject_mask, frame, covered,
        depth_map_valid=depth_map_valid, min_facing=min_facing,
    )


def view_texel_diagnostics_for_camera(
    positions: np.ndarray,
    normals: np.ndarray,
    camera: ProjectionCamera,
    depth_map: np.ndarray,
    depth_bias: float,
    subject_mask: np.ndarray,
    frame: tuple[float, float, float],
    covered: np.ndarray,
    depth_map_valid: np.ndarray | None = None,
    min_facing: float = 0.0,
) -> dict[str, int]:
    """Visibility funnel for one conditioning view, over `covered` atlas
    texels only: how many pass each successive test `view_texel_weights`
    applies, for diagnosing "why did this view contribute nothing" without
    re-deriving the same projection math by hand (the exact question the
    2026-09-04 GPU-box incident needed an answer to).

    Same arguments as `view_texel_weights_for_camera` (including
    `depth_map_valid` and `min_facing`), plus `covered` (the boolean atlas
    mask `rasterize_atlas` returns -- uncovered gutter texels are excluded
    from every count here).
    """
    masks = _view_texel_masks(
        positions, normals, camera, depth_map, depth_bias, subject_mask, frame,
        depth_map_valid=depth_map_valid, min_facing=min_facing,
    )
    covered_flat = np.asarray(covered, dtype=bool).reshape(-1)
    in_frame = masks["in_sample_bounds"]

    def count(mask: np.ndarray) -> int:
        return int((covered_flat & mask).sum())

    covered_depth = masks["depth"][covered_flat]
    return {
        "n_covered": int(covered_flat.sum()),
        "n_in_frame": count(in_frame),
        "n_depth_visible": count(masks["in_render_bounds"] & masks["within_depth"]),
        "n_in_subject_mask": count(in_frame & masks["subject"]),
        "n_facing_positive": count(masks["facing"] > 0.0),
        "n_weight_positive": count(masks["weights"] > 0.0),
        # For spotting a residual reference-frame offset or a genuinely
        # occluded role by eye (2026-09-04 GPU-box follow-up): the covered
        # texels' own analytic depth range, next to the whole render's.
        "depth_bias": float(depth_bias),
        "depth_map_min": masks["depth_map_min"],
        "depth_map_max": masks["depth_map_max"],
        "texel_depth_min": float(covered_depth.min()) if covered_depth.size else None,
        "texel_depth_max": float(covered_depth.max()) if covered_depth.size else None,
    }


def view_texel_debug_masks(
    positions: np.ndarray,
    normals: np.ndarray,
    basis: ViewBasis,
    center: np.ndarray,
    ortho_scale: float,
    image_size: int,
    depth_map: np.ndarray,
    depth_bias: float,
    subject_mask: np.ndarray,
    frame: tuple[float, float, float],
    camera_distance: float,
    depth_map_valid: np.ndarray | None = None,
    min_facing: float = 0.0,
) -> dict[str, np.ndarray]:
    """`view_texel_debug_masks_for_camera` for a square orthographic camera
    given as its parts."""
    return view_texel_debug_masks_for_camera(
        positions, normals,
        _orthographic_camera(basis, center, ortho_scale, image_size, camera_distance),
        depth_map, depth_bias, subject_mask, frame,
        depth_map_valid=depth_map_valid, min_facing=min_facing,
    )


def view_texel_debug_masks_for_camera(
    positions: np.ndarray,
    normals: np.ndarray,
    camera: ProjectionCamera,
    depth_map: np.ndarray,
    depth_bias: float,
    subject_mask: np.ndarray,
    frame: tuple[float, float, float],
    depth_map_valid: np.ndarray | None = None,
    min_facing: float = 0.0,
) -> dict[str, np.ndarray]:
    """The full-resolution (H, W) boolean/weight arrays behind
    `view_texel_weights`, exposed for debug visualization -- e.g. dumping
    per-role PNGs of which atlas texels pass which stage
    (AISMITH_PROJECT_VIEWS_DEBUG in blender_worker.py). Not on the normal
    projection path; `view_texel_weights`/`view_texel_diagnostics` compute
    (and discard) the same masks without ever exposing them.

    `in_mask` matches `view_texel_diagnostics`'s `n_in_subject_mask`
    convention (`in_frame & subject`, not `subject` alone): a subject-mask
    lookup outside the frame is a clipped, meaningless value.

    The keys beyond the original five are flat `(H*W,)` in atlas order (not
    reshaped) -- what the S2 projection conformance fixture needs to compare
    per-texel against the fixture's own ground truth without re-deriving
    `_view_texel_masks`' math a second time.
    """
    masks = _view_texel_masks(
        positions, normals, camera, depth_map, depth_bias, subject_mask, frame,
        depth_map_valid=depth_map_valid, min_facing=min_facing,
    )
    height, width = masks["height"], masks["width"]
    in_frame = masks["in_sample_bounds"]
    return {
        "in_frame": in_frame.reshape(height, width),
        "depth_visible": (masks["in_render_bounds"] & masks["within_depth"]).reshape(height, width),
        "in_mask": (in_frame & masks["subject"]).reshape(height, width),
        "facing_positive": (masks["facing"] > 0.0).reshape(height, width),
        "weights": masks["weights"].reshape(height, width),
        "depth": masks["depth"],
        "depth_at_render": masks["depth_at_render"],
        "render_px": masks["render_px"],
        "render_py": masks["render_py"],
        "render_ix": masks["render_ix"],
        "render_iy": masks["render_iy"],
        "sample_px": masks["sample_px"],
        "sample_py": masks["sample_py"],
        "facing": masks["facing"],
        "in_render_bounds": masks["in_render_bounds"],
        "within_depth": masks["within_depth"],
    }


def sample_bilinear(image_rgb: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Bilinear-sample an H x W x 3 image at fractional pixel coordinates,
    clamped to the image's edges."""
    image = np.asarray(image_rgb, dtype=np.float64)
    height, width = image.shape[:2]
    px = np.clip(np.asarray(px, dtype=np.float64), 0.0, width - 1.0)
    py = np.clip(np.asarray(py, dtype=np.float64), 0.0, height - 1.0)
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    fx = (px - x0)[..., None]
    fy = (py - y0)[..., None]
    top = image[y0, x0] * (1.0 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1.0 - fx) + image[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy


def blend_views(
    view_colors: list[np.ndarray],
    view_weights: list[np.ndarray],
    fallback_rgb: np.ndarray,
    min_total_weight: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted-average the views' colours per texel, falling back to the
    mesh's existing baked texture wherever no view carries enough weight."""
    fallback = np.asarray(fallback_rgb, dtype=np.float64)
    total_weight = np.zeros(fallback.shape[:2], dtype=np.float64)
    accumulated = np.zeros_like(fallback)
    for colors, weights in zip(view_colors, view_weights):
        colors = np.asarray(colors, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        accumulated += colors * weights[..., None]
        total_weight += weights
    projected_mask = total_weight >= min_total_weight
    safe_weight = np.where(projected_mask, total_weight, 1.0)
    blended = accumulated / safe_weight[..., None]
    result = np.where(projected_mask[..., None], blended, fallback)
    return result, projected_mask


def match_view_colour_to_reference(
    view_rgb: np.ndarray,
    view_weights: np.ndarray,
    reference_rgb: np.ndarray,
    reference_weights: np.ndarray,
    *,
    min_overlap: int = 500,
    gain_limits: tuple[float, float] = (0.9, 1.1),
    residual_ratio_limit: float = 0.7,
) -> tuple[np.ndarray, dict]:
    """Per-channel affine-match `view_rgb` onto `reference_rgb`'s statistics
    over the texels both views actually project to -- the fix for a side
    view's colour-drifted patch pulling the front's blend toward it (a
    brown-drifted coat greying out the front's black, 2026-09-05 GPU-box
    finding).

    `out = gain * in + offset`, `gain = std_ref / std_view` (clamped to
    `gain_limits` -- tight, because view drift is an offset, not a contrast
    change: a 0.5 gain flattened the right view on 2026-09-05), `offset =
    mean_ref - gain * mean_view`, all computed over
    texels where BOTH weight arrays are > 0 and applied to the WHOLE
    `view_rgb`. Too little overlap or a degenerate (zero-variance) view
    declines the match rather than dividing by ~zero.
    """
    view_rgb = np.asarray(view_rgb, dtype=np.float64)
    overlap = (np.asarray(view_weights) > 0.0) & (np.asarray(reference_weights) > 0.0)
    n_overlap = int(overlap.sum())
    if n_overlap < min_overlap:
        return view_rgb.copy(), {"applied": False, "overlap": n_overlap, "reason": "insufficient_overlap"}

    view_pixels = view_rgb[overlap]
    reference_pixels = np.asarray(reference_rgb, dtype=np.float64)[overlap]
    view_std = view_pixels.std(axis=0)
    if np.any(view_std <= 0.0):
        return view_rgb.copy(), {"applied": False, "overlap": n_overlap, "reason": "zero_variance"}

    gain = np.clip(reference_pixels.std(axis=0) / view_std, gain_limits[0], gain_limits[1])
    offset = reference_pixels.mean(axis=0) - gain * view_pixels.mean(axis=0)
    # A statistics match only means something when the overlap is the SAME
    # surface seen twice; on the conformance fixture the overlap sits on
    # colour-region boundaries and the fit is noise (gains 2.0 / 0.5 that
    # cost 18 points of classification accuracy). Apply only when the match
    # measurably reduces the overlap's own residual.
    before = float(np.mean(np.abs(view_pixels - reference_pixels)))
    after = float(np.mean(np.abs(np.clip(view_pixels * gain + offset, 0.0, 1.0) - reference_pixels)))
    if before <= 0.0 or after > residual_ratio_limit * before:
        return view_rgb.copy(), {
            "applied": False, "overlap": n_overlap, "reason": "no_residual_gain",
            "residual_before": round(before, 5), "residual_after": round(after, 5),
        }
    matched = np.clip(view_rgb * gain + offset, 0.0, 1.0)
    return matched, {
        "applied": True,
        "overlap": n_overlap,
        "gain": gain.tolist(),
        "offset": offset.tolist(),
        "residual_before": round(before, 5),
        "residual_after": round(after, 5),
    }


def dilate_projected(
    blended_rgb: np.ndarray,
    projected_mask: np.ndarray,
    fallback_rgb: np.ndarray,
    *,
    radius: int = DEFAULT_FILL_RADIUS,
    covered: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Grow `projected_mask` into unprojected texels instead of leaving them
    the reconstruction's blurred fallback bake (only ~30% of the atlas is
    ever projected, 2026-09-05 GPU-box finding).

    Each of `radius` passes gives every unprojected texel with >= 1 projected
    4/8-neighbour the mean colour of its projected neighbours and marks it
    projected for the next pass -- a vectorised binary dilation, not a
    per-pixel Python loop, so it stays fast on a 2048^2 atlas. A texel still
    unprojected after `radius` passes keeps `fallback_rgb`. `covered`, when
    given, confines growth to that mask (never spills into UV gutter).

    Returns `(rgb, filled_mask)`, where `filled_mask` marks texels that were
    NOT in `projected_mask` but got filled by this call.
    """
    rgb = np.asarray(blended_rgb, dtype=np.float64).copy()
    fallback = np.asarray(fallback_rgb, dtype=np.float64)
    projected = np.asarray(projected_mask, dtype=bool).copy()
    filled_mask = np.zeros_like(projected)
    growth_area = np.asarray(covered, dtype=bool) if covered is not None else None

    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    for _ in range(int(radius)):
        target = ~projected
        if growth_area is not None:
            target &= growth_area
        if not target.any():
            break
        color_sum = np.zeros_like(rgb)
        neighbor_count = np.zeros(projected.shape, dtype=np.float64)
        for row_offset, col_offset in offsets:
            shifted_valid = np.roll(projected, (row_offset, col_offset), axis=(0, 1))
            shifted_rgb = np.roll(rgb, (row_offset, col_offset), axis=(0, 1))
            if row_offset == -1:
                shifted_valid[-1, :] = False
            elif row_offset == 1:
                shifted_valid[0, :] = False
            if col_offset == -1:
                shifted_valid[:, -1] = False
            elif col_offset == 1:
                shifted_valid[:, 0] = False
            color_sum += shifted_rgb * shifted_valid[..., None]
            neighbor_count += shifted_valid
        newly_filled = target & (neighbor_count > 0)
        if not newly_filled.any():
            break
        rgb[newly_filled] = color_sum[newly_filled] / neighbor_count[newly_filled, None]
        projected[newly_filled] = True
        filled_mask[newly_filled] = True

    rgb = np.where(projected[..., None], rgb, fallback)
    return rgb, filled_mask


def color_delta_oklab(
    projected_rgb: np.ndarray,
    fallback_rgb: np.ndarray,
    mask: np.ndarray,
) -> float | None:
    """Mean OKLab distance between the blended texture and the prior bake,
    over `mask` (texels that carry both a projection and a fallback value).

    Both inputs are sRGB-ENCODED [0, 1] -- the space `image.pixels` hands
    back for a byte-buffer image (see `texture_bake.srgb_to_linear`'s
    docstring) -- so both are linearized before the perceptual comparison.
    `None` when `mask` selects no texels at all (nothing was projected).
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    projected = linear_rgb_to_oklab(srgb_to_linear(np.clip(np.asarray(projected_rgb, dtype=np.float64), 0.0, 1.0)))
    fallback = linear_rgb_to_oklab(srgb_to_linear(np.clip(np.asarray(fallback_rgb, dtype=np.float64), 0.0, 1.0)))
    delta = np.linalg.norm(projected[mask] - fallback[mask], axis=-1)
    return float(delta.mean())


def projection_report(
    per_role_coverage: dict[str, float],
    projected_mask: np.ndarray,
    covered: np.ndarray,
    occlusion_rejected_per_role: dict[str, float],
    delta_oklab_mean: float | None,
    atlas_size: int,
    frames: dict[str, tuple[float, float, float]],
) -> dict[str, Any]:
    """Provenance for one projection pass, recorded on the artifact."""
    projected_mask = np.asarray(projected_mask, dtype=bool)
    covered = np.asarray(covered, dtype=bool)
    covered_count = int(covered.sum())
    projected_count = int((projected_mask & covered).sum())
    fallback_count = covered_count - projected_count
    projected_fraction = projected_count / covered_count if covered_count else 0.0
    fallback_fraction = fallback_count / covered_count if covered_count else 0.0
    views_used = sorted(role for role, coverage in per_role_coverage.items() if coverage > 0.0)

    caveats: list[str] = []
    if covered_count == 0:
        caveats.append("The UV atlas has no covered texels; nothing could be projected.")
    elif projected_fraction < 0.5:
        caveats.append(
            "Less than half the atlas received a projected texel; most of the "
            "texture still comes from the existing bake."
        )
    if len(views_used) < len(per_role_coverage):
        unused = sorted(set(per_role_coverage) - set(views_used))
        caveats.append(f"View(s) contributed nothing: {', '.join(unused)}.")

    return {
        "schema_version": 1,
        "views_used": views_used,
        "per_role_coverage": {role: round(float(value), 6) for role, value in per_role_coverage.items()},
        "projected_texel_fraction": round(float(projected_fraction), 6),
        "fallback_texel_fraction": round(float(fallback_fraction), 6),
        "occlusion_rejected_fraction": {
            role: round(float(value), 6) for role, value in occlusion_rejected_per_role.items()
        },
        "mean_color_delta_vs_bake": (
            round(float(delta_oklab_mean), 6) if delta_oklab_mean is not None else None
        ),
        "atlas_size": int(atlas_size),
        "frame_alignment": {
            role: {"scale": float(scale), "dx": float(dx), "dy": float(dy)}
            for role, (scale, dx, dy) in frames.items()
        },
        "caveats": caveats,
    }


__all__ = [
    "ROLE_AZIMUTH_DEGREES",
    "EXTRA_ROLE_AZIMUTH_DEGREES",
    "DEFAULT_ROLE_PRIORITY",
    "DEFAULT_MIN_FACING",
    "DEFAULT_FILL_RADIUS",
    "BEHIND_CAMERA_PIXEL",
    "ViewBasis",
    "OrthographicCamera",
    "PerspectiveCamera",
    "ProjectionCamera",
    "ortho_view_basis",
    "azimuth_view_basis",
    "tilted_view_basis",
    "resolve_role_azimuth",
    "project_points",
    "project_points_perspective",
    "fit_frame_to_subject",
    "apply_frame",
    "invert_frame",
    "rasterize_atlas",
    "min_filter_neighborhood",
    "erode_mask",
    "head_bbox",
    "exclude_head",
    "mask_boundary_ring",
    "background_bleed_mask",
    "view_texel_weights",
    "view_texel_weights_for_camera",
    "view_texel_diagnostics",
    "view_texel_diagnostics_for_camera",
    "view_texel_debug_masks",
    "view_texel_debug_masks_for_camera",
    "sample_bilinear",
    "blend_views",
    "match_view_colour_to_reference",
    "dilate_projected",
    "color_delta_oklab",
    "projection_report",
]
