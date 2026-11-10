"""2D scorer for painted mesh-guided/idfirst views: scores a painted view
against the T-pose view of the SAME role directly in 2D (no mesh projection
involved) -- silhouette overlap, subject/face appearance similarity, detail
retention, and colour/luma drift, all from `compute_source_fidelity`.

Usage:
    python3 scripts/score_views_2d.py \\
        --pairs front=out/front.png:tpose/front.png,left=out/left.png:tpose/left.png \\
        --out out_dir
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# --- BEGIN vendored block -------------------------------------------------
# Copied from `git show feat/source-fidelity-gate:backend/app/source_fidelity.py`
# (that module does not exist on this branch; the branches merge later, so
# this is a private copy of only the pieces this scorer needs -- not a
# cross-branch import). Trimmed to `subject_mask_from_image`,
# `compute_source_fidelity`, and their helpers (camera-fitting and the
# side-by-side verdict-card image are not needed here).

POLICY_VERSION = "source-fidelity-v1"
FACE_BAND_FRACTION = 0.20
SSIM_WINDOW = 11
SSIM_SIGMA = 1.5
MS_SSIM_SCALES = 3
NEUTRAL_BACKGROUND = 0.5
MIN_FACE_PIXELS = 100
DETAIL_SIGMA = 1.5  # high-pass cutoff for detail_energy -- SSIM can't rank sharpness, this can
HEAD_BAND_TOP_FRACTION = 0.40  # columns must reach this far into the face band to count as head


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


def compute_source_fidelity(
    photo_rgba: np.ndarray,
    render_rgba: np.ndarray,
    *,
    photo_mask: np.ndarray | None = None,
    render_mask: np.ndarray | None = None,
    face_fraction: float = FACE_BAND_FRACTION,
    camera: Any | None = None,
    align: bool = True,
) -> dict[str, Any]:
    """Score a render against the photo it was generated from: silhouette
    overlap, appearance similarity over the region both agree is subject,
    appearance over the face specifically, a perceptual distance, and signed
    colour/luminance error -- everything a fidelity-gate policy needs."""
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

# --- END vendored block ----------------------------------------------------


TABLE_COLUMNS: list[tuple[str, str]] = [
    ("IoU", "silhouette_iou"),
    ("subject SSIM", "masked_ssim"),
    ("face SSIM", "face_ssim"),
    ("detail", "detail_ratio"),
    ("face detail", "face_detail_ratio"),
    ("luma Δ", "masked_luma_delta"),
    ("rgb L1", "masked_rgb_l1"),
]


def parse_pairs(raw: str) -> dict[str, tuple[Path, Path]]:
    """`role=painted.png:reference.png[,...]` -> {role: (painted, reference)}."""
    pairs: dict[str, tuple[Path, Path]] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        role, _, rest = entry.partition("=")
        painted, _, reference = rest.partition(":")
        pairs[role.strip()] = (Path(painted.strip()), Path(reference.strip()))
    return pairs


def load_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(opened.convert("RGBA"))


def _format_cell(value: float | None) -> str:
    return "--" if value is None else f"{value:.4f}"


def render_markdown_table(
    scores: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, float | None]]:
    header = "| role | " + " | ".join(name for name, _ in TABLE_COLUMNS) + " |"
    separator = "|" + "---|" * (len(TABLE_COLUMNS) + 1)
    lines = [header, separator]
    for role, metrics in scores.items():
        cells = [role] + [_format_cell(metrics.get(key)) for _, key in TABLE_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")

    means: dict[str, float | None] = {}
    for _, key in TABLE_COLUMNS:
        values = [metrics[key] for metrics in scores.values() if metrics.get(key) is not None]
        means[key] = round(sum(values) / len(values), 4) if values else None
    mean_cells = ["mean"] + [_format_cell(means[key]) for _, key in TABLE_COLUMNS]
    lines.append("| " + " | ".join(mean_cells) + " |")

    return "\n".join(lines), means


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = parse_pairs(args.pairs)
    args.out.mkdir(parents=True, exist_ok=True)

    scores: dict[str, dict[str, Any]] = {}
    for role, (painted_path, reference_path) in pairs.items():
        painted_rgba = load_rgba(painted_path)
        reference_rgba = load_rgba(reference_path)
        painted_mask, _ = subject_mask_from_image(painted_rgba)
        reference_mask, _ = subject_mask_from_image(reference_rgba)
        # This scores the painted VIEW against the T-pose view of the SAME
        # role -- identity in 2D, without projecting through the mesh.
        scores[role] = compute_source_fidelity(
            reference_rgba,
            painted_rgba,
            photo_mask=reference_mask,
            render_mask=painted_mask,
        )

    table, means = render_markdown_table(scores)
    print(table)

    (args.out / "scores.json").write_text(
        json.dumps({"scores": scores, "mean": means}, indent=2)
    )


if __name__ == "__main__":
    main()
