"""Source-fidelity metrics: score how precisely a rendered mesh reproduces
the ORIGINAL source photo it was generated from.

AISmith's governing requirement is precise match to the source image, not
plausibility -- a generated character can look fine and still have drifted
(a T-pose redraw against an arms-down photo, a shirt colour the texture head
invented). This module compares a photo and a same-framing render of the
mesh and reports silhouette overlap, appearance similarity over the subject
and over the face specifically, a perceptual distance, and signed colour/
luminance error, plus the 2D alignment applied and the camera spec used to
take the render -- everything a verdict-card policy needs to gate on.

The geometry metrics are pure numpy + PIL -- no scipy/skimage/torch/cv2 --
so they run in the same sandbox as the rest of the pipeline's policy checks;
the framing fit (`fit_source_camera`) is consumed by a Blender op, not this
module. The learned identity metrics (CLIP-I, DINOv2, LPIPS) are the one
exception: they need torch and are loaded lazily, and every one of them
degrades to `None` plus a `metric_warnings` entry when its weights are not
on the box, because this gate must never fail on a missing metric.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from PIL import Image

# PIL is imported inside the two functions that need it: the Blender worker
# imports this module for `fit_source_camera`, and Blender's bundled Python
# has no Pillow.

POLICY_VERSION = "source-fidelity-v1"
FACE_BAND_FRACTION = 0.20
SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
MS_SSIM_SCALES = 3
NEUTRAL_BACKGROUND = 0.5
MIN_FACE_PIXELS = 100
DETAIL_SIGMA = 1.5  # high-pass cutoff for detail_energy -- SSIM can't rank sharpness, this can
HEAD_BAND_TOP_FRACTION = 0.40  # columns must reach this far into the face band to count as head

# Identity metrics. SSIM measures shading, not identity: measured 2026-09-05,
# rendering ONE texture metallic vs dielectric moved masked_ssim by +-0.1, and
# SSIM could not separate a grey coat from a black one on identical geometry.
# CLIP-I, DINOv2 cosine and LPIPS are what the image-to-3D papers report
# instead (Hunyuan3D-Paint, Pixal3D, RomanTex) and key on the subject rather
# than on the lighting.
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
DINO_MODEL_ID = "facebook/dinov2-base"
LPIPS_NET = "vgg"
LPIPS_INPUT_SIZE = 224
# Weights come from the local caches only. A policy gate that silently pulls
# ~2GB the first time an operator runs it is worse than a missing metric, so
# fetching is opt-in.
IDENTITY_DOWNLOAD_ENV = "AISMITH_IDENTITY_ALLOW_DOWNLOAD"


def _as_float01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float64) / 255.0
    return arr.astype(np.float64)


def _to_uint8_rgba(rgba: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgba)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = np.round(arr).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    if arr.shape[-1] == 3:
        alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=-1)
    return arr


@dataclass(frozen=True)
class SourceCamera:
    """Camera the mesh is rendered from to reproduce the photo's framing.

    Conventions (must match view_projection.py): the pipeline's canonical
    "front" faces world -Y, world +Z is up; `yaw_degrees` is the azimuth
    about +Z measured from that front (front 0, left 90, back 180,
    right -90 -- same table as view_projection.ROLE_AZIMUTH_DEGREES);
    `pitch_degrees` > 0 looks down at the subject. The camera sits at
    `center + direction * distance` looking at `center`.
    `fov_x_degrees` is the HORIZONTAL field of view (perspective only);
    `ortho_scale` is world units across the image WIDTH (orthographic only).
    """

    kind: Literal["perspective", "orthographic"]
    width: int
    height: int
    center: tuple[float, float, float]
    distance: float
    yaw_degrees: float = 0.0
    pitch_degrees: float = 0.0
    fov_x_degrees: float | None = None
    ortho_scale: float | None = None

    def direction(self) -> np.ndarray:
        """Unit vector from `center` toward the camera; matches
        `ortho_view_basis` at pitch 0 and adds elevation for pitch."""
        yaw = math.radians(self.yaw_degrees)
        pitch = math.radians(self.pitch_degrees)
        return np.array(
            [
                math.sin(yaw) * math.cos(pitch),
                -math.cos(yaw) * math.cos(pitch),
                math.sin(pitch),
            ],
            dtype=np.float64,
        )

    def to_spec(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "width": self.width,
            "height": self.height,
            "center": list(self.center),
            "distance": self.distance,
            "yaw_degrees": self.yaw_degrees,
            "pitch_degrees": self.pitch_degrees,
            "fov_x_degrees": self.fov_x_degrees,
            "ortho_scale": self.ortho_scale,
        }

    @staticmethod
    def from_spec(spec: dict[str, Any]) -> "SourceCamera":
        return SourceCamera(
            kind=spec["kind"],
            width=spec["width"],
            height=spec["height"],
            center=tuple(spec["center"]),
            distance=spec["distance"],
            yaw_degrees=spec.get("yaw_degrees", 0.0),
            pitch_degrees=spec.get("pitch_degrees", 0.0),
            fov_x_degrees=spec.get("fov_x_degrees"),
            ortho_scale=spec.get("ortho_scale"),
        )


def fit_source_camera(
    mesh_low,
    mesh_high,
    image_size: tuple[int, int],
    subject_bbox: tuple[int, int, int, int],
    *,
    fov_x_degrees: float | None = None,
    yaw_degrees: float = 0.0,
    pitch_degrees: float = 0.0,
) -> SourceCamera:
    """Frame the mesh so its vertical extent fills the same fraction of the
    image height the subject bbox fills in the photo, and its horizontal
    extent fills at most the bbox's fraction of the width -- whichever
    constraint is tighter wins, since yaw is arbitrary and either extent
    could be the one facing the camera."""
    mesh_low = np.asarray(mesh_low, dtype=np.float64)
    mesh_high = np.asarray(mesh_high, dtype=np.float64)
    width, height = image_size
    x0, y0, x1, y1 = subject_bbox
    fill_h = (y1 - y0) / height
    fill_w = (x1 - x0) / width
    center = tuple(((mesh_low + mesh_high) / 2.0).tolist())
    mesh_height = float(mesh_high[2] - mesh_low[2])
    mesh_width = float(max(mesh_high[0] - mesh_low[0], mesh_high[1] - mesh_low[1]))

    if fov_x_degrees is not None:
        half_fov = math.radians(fov_x_degrees) / 2.0
        distance_h = mesh_height / fill_h / (2 * math.tan(half_fov) * (height / width))
        distance_w = mesh_width / fill_w / (2 * math.tan(half_fov))
        distance = max(distance_h, distance_w)
        return SourceCamera(
            kind="perspective",
            width=width,
            height=height,
            center=center,
            distance=distance,
            yaw_degrees=yaw_degrees,
            pitch_degrees=pitch_degrees,
            fov_x_degrees=fov_x_degrees,
            ortho_scale=None,
        )

    ortho_scale = max(mesh_height / fill_h * (width / height), mesh_width / fill_w)
    diagonal = float(np.linalg.norm(mesh_high - mesh_low))
    return SourceCamera(
        kind="orthographic",
        width=width,
        height=height,
        center=center,
        distance=2 * diagonal,
        yaw_degrees=yaw_degrees,
        pitch_degrees=pitch_degrees,
        fov_x_degrees=None,
        ortho_scale=ortho_scale,
    )


def subject_mask_from_image(rgba: np.ndarray) -> tuple[np.ndarray, str]:
    """Segment the subject from its background. Alpha is authoritative when
    present; otherwise the 2-pixel border ring stands in for the background
    colour and anything far enough from it is the subject."""
    arr = np.asarray(rgba)
    if arr.shape[-1] == 4:
        alpha = _as_float01(arr[..., 3])
        if np.any(alpha < 128 / 255.0):
            return alpha >= 128 / 255.0, "alpha"
    rgb = _as_float01(arr[..., :3])
    ring = np.concatenate(
        [
            rgb[:2].reshape(-1, 3),
            rgb[-2:].reshape(-1, 3),
            rgb[:, :2].reshape(-1, 3),
            rgb[:, -2:].reshape(-1, 3),
        ]
    )
    border_colour = np.median(ring, axis=0)
    distance = np.linalg.norm(rgb - border_colour, axis=-1)
    return distance > 0.12, "border-colour"


def subject_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) with x1/y1 exclusive; raises on an empty mask since
    there is no bbox to return."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("subject_bbox: mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def face_band_bbox(bbox, fraction: float = FACE_BAND_FRACTION) -> tuple[int, int, int, int]:
    """The top `fraction` of the bbox height, same x range, at least one row."""
    x0, y0, x1, y1 = bbox
    band_height = max(1, round((y1 - y0) * fraction))
    return x0, y0, x1, min(y1, y0 + band_height)


def head_bbox(mask: np.ndarray, fraction: float = FACE_BAND_FRACTION) -> tuple[int, int, int, int]:
    """Head-only box within the face band: on a T-pose the outstretched arms
    fill the band too, but they enter it from BELOW while the head enters
    from ABOVE, so keep only the columns that reach into the band's own top
    40% and take the mask's bbox over just those columns. Falls back to the
    plain `face_band_bbox` when no column qualifies (e.g. an empty mask)."""
    mask = np.asarray(mask, dtype=bool)
    band = face_band_bbox(subject_bbox(mask), fraction)
    bx0, by0, bx1, by1 = band
    top_cut = by0 + max(1, round((by1 - by0) * HEAD_BAND_TOP_FRACTION))
    top_slice = mask[by0:top_cut, bx0:bx1]
    qualifying = np.nonzero(top_slice.any(axis=0))[0]
    if qualifying.size == 0:
        return band
    col0, col1 = bx0 + int(qualifying.min()), bx0 + int(qualifying.max()) + 1
    band_slice = mask[by0:by1, col0:col1]
    ys, xs = np.nonzero(band_slice)
    if ys.size == 0:
        return band
    return col0 + int(xs.min()), by0 + int(ys.min()), col0 + int(xs.max()) + 1, by0 + int(ys.max()) + 1


def composite_over_grey(rgba: np.ndarray, grey: float = NEUTRAL_BACKGROUND) -> np.ndarray:
    """Straight-alpha composite over flat grey; float32 HxWx3 in [0,1] so
    transparent pixels don't bias appearance comparisons toward black."""
    arr = _as_float01(np.asarray(rgba))
    rgb = arr[..., :3]
    if arr.shape[-1] == 4:
        alpha = arr[..., 3:4]
    else:
        alpha = np.ones(rgb.shape[:2] + (1,), dtype=np.float64)
    return (rgb * alpha + grey * (1 - alpha)).astype(np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _gaussian_kernel(sigma: float, window: int) -> np.ndarray:
    radius = window // 2
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_blur(image: np.ndarray, sigma: float = SSIM_SIGMA, window: int = SSIM_WINDOW) -> np.ndarray:
    """Separable Gaussian blur, reflect-padded, vectorised via sliding
    windows -- no Python loop over pixels, so it stays fast on full-res
    renders."""
    kernel = _gaussian_kernel(sigma, window)
    radius = window // 2
    padded = np.pad(np.asarray(image, dtype=np.float64), radius, mode="reflect")
    windows_h = np.lib.stride_tricks.sliding_window_view(padded, window, axis=1)
    horiz = np.tensordot(windows_h, kernel, axes=([-1], [0]))
    windows_v = np.lib.stride_tricks.sliding_window_view(horiz, window, axis=0)
    return np.tensordot(windows_v, kernel, axes=([-1], [0]))


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Standard SSIM (Wang 2004): L=1, C1=(0.01)^2, C2=(0.03)^2, gaussian
    window, for 2-D [0,1] float arrays."""
    c1 = 0.01**2
    c2 = 0.03**2
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mu_a = gaussian_blur(a)
    mu_b = gaussian_blur(b)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_a2 = gaussian_blur(a * a) - mu_a2
    sigma_b2 = gaussian_blur(b * b) - mu_b2
    sigma_ab = gaussian_blur(a * b) - mu_ab
    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    return numerator / denominator


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    return float(np.mean(values[mask]))


def _erode_square(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    window = 2 * radius + 1
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    view = np.lib.stride_tricks.sliding_window_view(padded, (window, window))
    return view.all(axis=(-2, -1))


def masked_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
    """Mean SSIM over `mask`, eroded by half the SSIM window so no pixel
    whose blur window straddles the mask boundary contaminates the score."""
    eroded = _erode_square(np.asarray(mask, dtype=bool), SSIM_WINDOW // 2)
    if not np.any(eroded):
        return None
    return masked_mean(ssim_map(a, b), eroded)


def detail_energy(gray: np.ndarray, mask: np.ndarray) -> float | None:
    """Mean high-pass energy over `mask`: SSIM rewards local structure match
    and can score a sharp-but-slightly-misregistered render BELOW a blurred
    one, so a fidelity gate needs a separate sharpness signal. None when the
    eroded mask has nothing left to measure."""
    eroded = _erode_square(np.asarray(mask, dtype=bool), SSIM_WINDOW // 2)
    if not np.any(eroded):
        return None
    gray = np.asarray(gray, dtype=np.float64)
    high_pass = np.abs(gray - gaussian_blur(gray, sigma=DETAIL_SIGMA))
    return masked_mean(high_pass, eroded)


def _downsample2x(a: np.ndarray) -> np.ndarray:
    h2, w2 = a.shape[0] // 2, a.shape[1] // 2
    a = a[: h2 * 2, : w2 * 2]
    return a.reshape(h2, 2, w2, 2).mean(axis=(1, 3))


def _downsample2x_mask(mask: np.ndarray) -> np.ndarray:
    h2, w2 = mask.shape[0] // 2, mask.shape[1] // 2
    mask = mask[: h2 * 2, : w2 * 2]
    return mask.reshape(h2, 2, w2, 2).all(axis=(1, 3))


def ms_ssim_proxy(a: np.ndarray, b: np.ndarray, mask: np.ndarray, scales: int = MS_SSIM_SCALES) -> float | None:
    """Numpy-only multi-scale SSIM stand-in: product of masked SSIM across
    `scales` levels of 2x2 box downsampling, stopping early once the image
    would go below the SSIM window; None if even the first scale is None."""
    product = None
    cur_a, cur_b, cur_mask = a, b, mask
    for _ in range(scales):
        if min(cur_a.shape[0], cur_a.shape[1]) < SSIM_WINDOW:
            break
        value = masked_ssim(cur_a, cur_b, cur_mask)
        if value is None:
            break
        product = value if product is None else product * value
        cur_a = _downsample2x(cur_a)
        cur_b = _downsample2x(cur_b)
        cur_mask = _downsample2x_mask(cur_mask)
    return product


def silhouette_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU of two boolean subject masks; 0.0 when the union is empty rather
    than a NaN from a 0/0 division."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def align_render_to_photo(
    render_rgba: np.ndarray, render_mask: np.ndarray, photo_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Uniform-scale + translate the render onto the photo's frame so the
    subject bboxes match in height and centre. `bbox_width_ratio` is the
    residual width mismatch this can't fix -- a pose/shape difference (e.g.
    a T-pose against arms-down) shows up there, not in the shape it's given."""
    from PIL import Image

    photo_box = subject_bbox(photo_mask)
    render_box = subject_bbox(render_mask)
    px0, py0, px1, py1 = photo_box
    rx0, ry0, rx1, ry1 = render_box
    scale = (py1 - py0) / (ry1 - ry0)
    photo_cx, photo_cy = (px0 + px1) / 2.0, (py0 + py1) / 2.0
    render_cx, render_cy = (rx0 + rx1) / 2.0, (ry0 + ry1) / 2.0
    height, width = photo_mask.shape

    a = 1.0 / scale
    c = render_cx - photo_cx / scale
    e = 1.0 / scale
    f = render_cy - photo_cy / scale
    matrix = (a, 0.0, c, 0.0, e, f)

    render_img = Image.fromarray(_to_uint8_rgba(render_rgba), mode="RGBA")
    aligned_img = render_img.transform(
        (width, height), Image.AFFINE, matrix, resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0)
    )
    aligned_render = np.asarray(aligned_img)

    mask_img = Image.fromarray((np.asarray(render_mask, dtype=np.uint8) * 255), mode="L")
    aligned_mask_img = mask_img.transform(
        (width, height), Image.AFFINE, matrix, resample=Image.NEAREST, fillcolor=0
    )
    aligned_mask = np.asarray(aligned_mask_img) > 127

    render_w_after = (rx1 - rx0) * scale
    photo_w = px1 - px0
    info = {
        "scale": scale,
        "dx": photo_cx - scale * render_cx,
        "dy": photo_cy - scale * render_cy,
        "render_bbox": list(render_box),
        "photo_bbox": list(photo_box),
        "bbox_width_ratio": render_w_after / photo_w,
    }
    return aligned_render, aligned_mask, info


@dataclass
class IdentityModels:
    """The three identity backbones, loaded once per process by
    `load_identity_models`. Each one is independent: transformers missing
    leaves LPIPS working, and vice versa, so a partial install still buys a
    partial answer. `warnings` explains every backbone that is `None`."""

    device: str = "cpu"
    torch: Any = None
    clip: Any = None
    clip_processor: Any = None
    dino: Any = None
    dino_processor: Any = None
    lpips: Any = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


_IDENTITY_MODELS: IdentityModels | None = None


def _download_allowed() -> bool:
    return os.environ.get(IDENTITY_DOWNLOAD_ENV, "").strip().lower() not in ("", "0", "false", "no")


def _torch_hub_has_vgg(torch: Any) -> bool:
    """LPIPS pulls torchvision's pretrained VGG through torch.hub, which has
    no `local_files_only` switch -- so check the cache ourselves instead."""
    checkpoints = Path(torch.hub.get_dir()) / "checkpoints"
    return checkpoints.is_dir() and any(checkpoints.glob("vgg16-*.pth"))


def _load_identity_models() -> IdentityModels:
    # Every except here is deliberately bare: a missing wheel, a cache miss, a
    # CUDA mismatch and a corrupt checkpoint all mean the same thing to this
    # gate -- "no metric" -- and none of them may take a fidelity run down.
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return IdentityModels(warnings=(f"clip_i/dino_cos/lpips unavailable: torch not importable ({exc})",))

    warnings: list[str] = []
    local_only = not _download_allowed()
    models = IdentityModels(device="cuda" if torch.cuda.is_available() else "cpu", torch=torch)

    try:
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"clip_i/dino_cos unavailable: transformers not importable ({exc})")
    else:
        try:
            models.clip = CLIPModel.from_pretrained(CLIP_MODEL_ID, local_files_only=local_only).to(models.device).eval()
            models.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, local_files_only=local_only)
        except Exception as exc:  # noqa: BLE001
            models.clip, models.clip_processor = None, None
            warnings.append(f"clip_i unavailable: {CLIP_MODEL_ID} not loadable ({exc})")
        try:
            models.dino = AutoModel.from_pretrained(DINO_MODEL_ID, local_files_only=local_only).to(models.device).eval()
            models.dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID, local_files_only=local_only)
        except Exception as exc:  # noqa: BLE001
            models.dino, models.dino_processor = None, None
            warnings.append(f"dino_cos unavailable: {DINO_MODEL_ID} not loadable ({exc})")

    try:
        import lpips as lpips_package
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"lpips unavailable: lpips not importable ({exc})")
    else:
        if local_only and not _torch_hub_has_vgg(torch):
            warnings.append(
                f"lpips unavailable: torchvision {LPIPS_NET} weights are not in the torch hub cache "
                f"(set {IDENTITY_DOWNLOAD_ENV}=1 to fetch them)"
            )
        else:
            try:
                models.lpips = lpips_package.LPIPS(net=LPIPS_NET, verbose=False).to(models.device).eval()
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"lpips unavailable: {exc}")

    models.warnings = tuple(warnings)
    return models


def load_identity_models(*, reload: bool = False) -> IdentityModels:
    """Process-wide singleton: the weights are hundreds of MB, and a failed
    load is cached too so a box without them doesn't retry per view."""
    global _IDENTITY_MODELS
    if _IDENTITY_MODELS is None or reload:
        _IDENTITY_MODELS = _load_identity_models()
    return _IDENTITY_MODELS


def _to_pil(image: np.ndarray) -> Image.Image:
    from PIL import Image

    pixels = np.round(np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _clip_embedding(models: IdentityModels, image: np.ndarray) -> np.ndarray:
    """CLIP ViT-L/14 image embedding; the processor owns the 224px resize and
    the CLIP normalisation, so callers hand over full-size crops."""
    inputs = models.clip_processor(images=_to_pil(image), return_tensors="pt").to(models.device)
    with models.torch.no_grad():
        features = models.clip.get_image_features(**inputs)
    return features[0].float().cpu().numpy()


def _dino_embedding(models: IdentityModels, image: np.ndarray) -> np.ndarray:
    """DINOv2 CLS token -- the patch tokens are position-bound, the CLS is the
    image-level identity vector the papers compare."""
    inputs = models.dino_processor(images=_to_pil(image), return_tensors="pt").to(models.device)
    with models.torch.no_grad():
        outputs = models.dino(**inputs)
    return outputs.last_hidden_state[0, 0].float().cpu().numpy()


def _lpips_forward(models: IdentityModels, a: np.ndarray, b: np.ndarray) -> float:
    torch = models.torch
    tensors = []
    for image in (a, b):
        array = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)[None]
        tensor = torch.nn.functional.interpolate(
            tensor, size=(LPIPS_INPUT_SIZE, LPIPS_INPUT_SIZE), mode="bilinear", align_corners=False
        )
        tensors.append(tensor.to(models.device) * 2.0 - 1.0)  # LPIPS expects [-1, 1]
    with torch.no_grad():
        distance = models.lpips(tensors[0], tensors[1])
    return float(distance.reshape(-1)[0].cpu())


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))


def clip_similarity(a: np.ndarray, b: np.ndarray, *, models: IdentityModels | None = None) -> float | None:
    """CLIP-I: cosine of two CLIP image embeddings, in [-1, 1], 1 = identical.
    `None` when the model is not on the box."""
    models = load_identity_models() if models is None else models
    if models.clip is None:
        return None
    return _cosine(_clip_embedding(models, a), _clip_embedding(models, b))


def dino_similarity(a: np.ndarray, b: np.ndarray, *, models: IdentityModels | None = None) -> float | None:
    """DINOv2 CLS cosine, in [-1, 1], 1 = identical. `None` when the model is
    not on the box."""
    models = load_identity_models() if models is None else models
    if models.dino is None:
        return None
    return _cosine(_dino_embedding(models, a), _dino_embedding(models, b))


def lpips_distance(a: np.ndarray, b: np.ndarray, *, models: IdentityModels | None = None) -> float | None:
    """LPIPS perceptual distance, ~[0, 1], 0 = identical -- the only one of
    the three where LOWER is better. `None` when the model is not on the box."""
    models = load_identity_models() if models is None else models
    if models.lpips is None:
        return None
    return _lpips_forward(models, a, b)


def masked_crop(
    rgb: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int], *, grey: float = NEUTRAL_BACKGROUND
) -> np.ndarray:
    """The exact pixels masked_ssim/face_ssim score, as a standalone image:
    `box` cropped out of the grey-composited frame with everything outside
    `mask` flattened to that same grey, so a learned metric sees the subject
    and not the background it happens to sit on."""
    x0, y0, x1, y1 = box
    crop = np.array(rgb[y0:y1, x0:x1], dtype=np.float32)
    crop[~np.asarray(mask, dtype=bool)[y0:y1, x0:x1]] = grey
    return crop


IDENTITY_METRIC_KEYS = ("clip_i", "clip_i_face", "dino_cos", "dino_cos_face", "lpips", "lpips_face")


def identity_metrics(
    photo_rgb: np.ndarray,
    render_rgb: np.ndarray,
    region: np.ndarray,
    subject_box: tuple[int, int, int, int],
    face_region: np.ndarray,
    face_box: tuple[int, int, int, int],
    *,
    models: IdentityModels | None = None,
) -> tuple[dict[str, float | None], list[str]]:
    """CLIP-I / DINOv2 cosine / LPIPS over the SAME crops the SSIM numbers
    use, so every value in the report describes the same pixels. Returns the
    values and the warnings explaining any that are `None`."""
    models = load_identity_models() if models is None else models
    values: dict[str, float | None] = dict.fromkeys(IDENTITY_METRIC_KEYS)

    def _measure(mask: np.ndarray, box: tuple[int, int, int, int]) -> tuple[float | None, float | None, float | None]:
        x0, y0, x1, y1 = box
        if x1 - x0 < 1 or y1 - y0 < 1:
            return None, None, None
        a = masked_crop(photo_rgb, mask, box)
        b = masked_crop(render_rgb, mask, box)
        return (
            clip_similarity(a, b, models=models),
            dino_similarity(a, b, models=models),
            lpips_distance(a, b, models=models),
        )

    values["clip_i"], values["dino_cos"], values["lpips"] = _measure(region, subject_box)
    # Same floor face_ssim uses -- below it the crop is too small to mean anything.
    if int(np.count_nonzero(face_region)) >= MIN_FACE_PIXELS:
        values["clip_i_face"], values["dino_cos_face"], values["lpips_face"] = _measure(face_region, face_box)
    return values, list(models.warnings)


def compute_source_fidelity(
    photo_rgba: np.ndarray,
    render_rgba: np.ndarray,
    *,
    photo_mask: np.ndarray | None = None,
    render_mask: np.ndarray | None = None,
    face_fraction: float = FACE_BAND_FRACTION,
    camera: SourceCamera | None = None,
    align: bool = True,
) -> dict[str, Any]:
    """Score a render against the photo it was generated from: silhouette
    overlap, appearance similarity over the region both agree is subject,
    appearance over the face specifically, a perceptual distance, signed
    colour/luminance error, and the learned identity metrics (CLIP-I, DINOv2
    cosine, LPIPS) over those same two crops -- everything a fidelity-gate
    policy needs. The identity metrics come back `None` with an explanation
    in `metric_warnings` on a box without the weights."""
    photo_rgba = np.asarray(photo_rgba)
    render_rgba = np.asarray(render_rgba)

    if photo_mask is None:
        photo_mask, mask_source = subject_mask_from_image(photo_rgba)
    else:
        photo_mask = np.asarray(photo_mask, dtype=bool)
        mask_source = "provided"
    if render_mask is None:
        render_mask, _ = subject_mask_from_image(render_rgba)
    else:
        render_mask = np.asarray(render_mask, dtype=bool)

    if align:
        aligned_render, aligned_render_mask, alignment = align_render_to_photo(
            render_rgba, render_mask, photo_mask
        )
    else:
        if render_rgba.shape[:2] != photo_rgba.shape[:2]:
            raise ValueError("compute_source_fidelity: shapes must match when align=False")
        aligned_render, aligned_render_mask, alignment = render_rgba, render_mask, None

    region = photo_mask & aligned_render_mask
    photo_rgb = composite_over_grey(photo_rgba)
    render_rgb = composite_over_grey(aligned_render)
    photo_luma = luminance(photo_rgb)
    render_luma = luminance(render_rgb)

    masked_ssim_value = masked_ssim(photo_luma, render_luma, region)
    ms_ssim_value = ms_ssim_proxy(photo_luma, render_luma, region)
    perceptual_distance = None if ms_ssim_value is None else 1.0 - ms_ssim_value

    subject_box = subject_bbox(photo_mask)
    face_box = head_bbox(photo_mask, face_fraction)
    fx0, fy0, fx1, fy1 = face_box
    face_region = np.zeros_like(region)
    face_region[fy0:fy1, fx0:fx1] = region[fy0:fy1, fx0:fx1]
    face_pixels = int(np.count_nonzero(face_region))
    face_ssim_value = masked_ssim(photo_luma, render_luma, face_region) if face_pixels >= MIN_FACE_PIXELS else None

    luma_delta = masked_mean(render_luma - photo_luma, region)
    rgb_l1 = masked_mean(np.mean(np.abs(render_rgb - photo_rgb), axis=-1), region)
    iou = silhouette_iou(photo_mask, aligned_render_mask)

    photo_detail = detail_energy(photo_luma, region)
    render_detail = detail_energy(render_luma, region)
    photo_face_detail = detail_energy(photo_luma, face_region)
    render_face_detail = detail_energy(render_luma, face_region)

    def _ratio(numerator, denominator):
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator

    detail_ratio = _ratio(render_detail, photo_detail)
    face_detail_ratio = _ratio(render_face_detail, photo_face_detail)

    identity, metric_warnings = identity_metrics(
        photo_rgb, render_rgb, region, subject_box, face_region, face_box
    )

    def _round(value):
        return None if value is None else round(float(value), 4)

    return {
        "policy_version": POLICY_VERSION,
        "silhouette_iou": _round(iou),
        "masked_ssim": _round(masked_ssim_value),
        "face_ssim": _round(face_ssim_value),
        "perceptual_distance": _round(perceptual_distance),
        "perceptual_metric": "ms_ssim_proxy",
        "masked_luma_delta": _round(luma_delta),
        "masked_rgb_l1": _round(rgb_l1),
        "photo_detail": _round(photo_detail),
        "render_detail": _round(render_detail),
        "detail_ratio": _round(detail_ratio),
        "face_detail_ratio": _round(face_detail_ratio),
        **{key: _round(value) for key, value in identity.items()},
        "identity_models": {"clip": CLIP_MODEL_ID, "dino": DINO_MODEL_ID, "lpips": LPIPS_NET},
        "metric_warnings": metric_warnings,
        "subject_bbox": list(subject_box),
        "face_bbox": list(face_box),
        "face_pixels": face_pixels,
        "region_pixels": int(np.count_nonzero(region)),
        "mask_source": mask_source,
        "alignment": alignment,
        "camera": camera.to_spec() if camera else None,
        "photo_size": [photo_rgba.shape[1], photo_rgba.shape[0]],
        "render_size": [render_rgba.shape[1], render_rgba.shape[0]],
    }


def side_by_side(
    photo_rgba: np.ndarray, aligned_render_rgba: np.ndarray, face_bbox, *, grey: float = NEUTRAL_BACKGROUND
) -> "Image.Image":
    """Verdict-card composite: photo | render | absolute luminance
    difference (white = big difference), with the face band boxed on the
    first two panels."""
    from PIL import Image, ImageDraw

    photo_rgb = composite_over_grey(photo_rgba, grey)
    render_rgb = composite_over_grey(aligned_render_rgba, grey)
    height, width = photo_rgb.shape[:2]
    diff = np.abs(luminance(render_rgb) - luminance(photo_rgb))
    diff_rgb = np.stack([diff, diff, diff], axis=-1)

    def _to_image(arr: np.ndarray) -> Image.Image:
        pixels = np.round(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        return Image.fromarray(pixels, mode="RGB")

    photo_img = _to_image(photo_rgb)
    render_img = _to_image(render_rgb)
    diff_img = _to_image(diff_rgb)

    x0, y0, x1, y1 = face_bbox
    for panel in (photo_img, render_img):
        ImageDraw.Draw(panel).rectangle([x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)], outline=(255, 0, 255), width=2)

    canvas = Image.new("RGB", (3 * width, height))
    canvas.paste(photo_img, (0, 0))
    canvas.paste(render_img, (width, 0))
    canvas.paste(diff_img, (2 * width, 0))
    return canvas
