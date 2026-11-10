"""Pure helpers for sizing, packing and repairing selected-to-active bakes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

# The canonicalize bake atlas is derived from the SOURCE's largest texture
# rather than pinned to a constant, so a 4K-textured generate result is not
# silently resampled down (2026-09-01 P1: a hardcoded 2048 atlas combined with
# a 6.8%-utilised UV layout gave the canonical mesh ~645 texels per world-unit
# against the S1 source's ~1,146 -- the visibly muddy face the operator
# caught by eye while the S1/S2 appearance gate still scored 0.997).
BAKE_TEXTURE_SIZE_MINIMUM = 1024
# 4096^2 x RGBA float is already ~256 MB of Blender image buffer per channel;
# above this the bake stops paying for itself in wall time and memory.
BAKE_TEXTURE_SIZE_MAXIMUM = 4096
# Gap left between packed UV islands, measured in texels of the bake atlas.
# Blender's `pack_islands` margin is a fraction of the UV unit square, so it
# only means a fixed number of texels once it is derived from the atlas size.
# Measured on the alien's canonical mesh (~2,000 smart-project islands): 4
# texels packs to ~46% atlas utilisation, 6 texels to ~40%, 10 to ~26%.
UV_PACK_MARGIN_TEXELS = 4

# Every bake image the canonicalize pass writes shares this prefix so the
# exported GLB's texture names say where they came from.
BAKE_IMAGE_PREFIX = "AISmith_Retopo_"
# A tangent-space normal encoded in [0, 1]: (0, 0, 1) -> (0.5, 0.5, 1.0), i.e.
# "no deviation from the interpolated surface normal". Texels the bake never
# reaches keep this, so a bake miss reads as untouched shading rather than as
# a hole punched through the lighting.
NORMAL_MAP_FLAT_RGB = (0.5, 0.5, 1.0)
# glTF 2.0 requires OpenGL-convention tangent-space normal maps: +X right,
# +Y UP (green channel points up), +Z out of the surface. That is Blender's
# own bake default (`normal_r/g/b = POS_X/POS_Y/POS_Z`), so the two agree --
# but the pairing is exactly the trap that silently inverts a normal map's
# lighting, so it is asserted in the report rather than assumed.
NORMAL_MAP_GREEN_CHANNEL = "+Y"
NORMAL_MAP_SPACE = "tangent"
NORMAL_MAP_COLORSPACE = "Non-Color"
# Tolerance for "this texel is unchanged from its pre-bake value", i.e. the
# bake never wrote it. `bpy.data.images.new` allocates a BYTE buffer, so the
# smallest representable change is 1/255 ~= 0.0039 and anything below this
# tolerance is genuinely the same texel.
NORMAL_MAP_UNTOUCHED_TOLERANCE = 1e-4


def bake_image_name(channel: str, *, prefix: str = BAKE_IMAGE_PREFIX) -> str:
    """Datablock name for one baked channel image (`BaseColor`, `Normal`, ...).

    Names are stable and channel-derived because they survive into the
    exported GLB's texture list, where they are the only clue about which
    pass produced a given image.
    """
    cleaned = "".join(part for part in str(channel).split())
    if not cleaned:
        raise ValueError(f"bake_image_name needs a channel name, got {channel!r}")
    return f"{prefix}{cleaned}"


def mark_normal_bake_coverage(
    pixels: np.ndarray,
    before: np.ndarray,
    *,
    tolerance: float = NORMAL_MAP_UNTOUCHED_TOLERANCE,
) -> np.ndarray:
    """Rewrite alpha to mark which texels Blender's NORMAL bake actually wrote.

    The emit passes get their coverage for free: Cycles leaves a bake miss at
    the image's transparent generated colour, so alpha < 0.5 IS the miss mask
    that `fill_transparent_texels` dilates into. The NORMAL pass has no such
    contract -- it is a geometry pass and its alpha is not a coverage channel
    -- so coverage is measured by DIFFING the atlas against its own pre-bake
    contents: an RGB triple the bake did not move is a texel the bake missed.

    Comparing against the nominal flat constant instead does not work.
    `bpy.data.images.new` allocates a byte buffer, so a `generated_color` of
    0.5 comes back as 127/255 = 0.498039 and every texel reads as "changed"
    -- which is exactly how this measurement first reported a 0.00% miss rate
    on an atlas only 42% covered by UV islands (2026-09-01, alien).

    A genuinely-baked texel that lands on its own pre-bake value is counted
    as a miss, which is harmless: dilating a neighbour into it and leaving it
    alone produce the same shading.
    """
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("Normal bake pixels must be an H x W x 4 RGBA array")
    baked = np.asarray(pixels, dtype=np.float32)
    original = np.asarray(before, dtype=np.float32)
    if original.shape != baked.shape:
        raise ValueError(
            f"Normal bake coverage needs matching buffers, got {baked.shape} vs {original.shape}"
        )
    marked = baked.copy()
    untouched = np.all(
        np.abs(baked[:, :, :3] - original[:, :, :3]) <= tolerance, axis=2,
    )
    marked[:, :, 3] = np.where(untouched, 0.0, 1.0)
    return marked


def renormalize_tangent_normals(pixels: np.ndarray) -> np.ndarray:
    """Re-unit-length every encoded normal after the miss dilation.

    `fill_transparent_texels` averages neighbouring RGBA, which is the right
    thing for colour and the wrong thing for a direction: the mean of two unit
    vectors is shorter than one, and a shortened encoded normal reads to the
    shader as a normal tilted toward the surface. Decode, normalise, re-encode.
    A degenerate (zero-length) sample falls back to the flat normal rather
    than dividing by zero.
    """
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("Normal map pixels must be an H x W x 4 RGBA array")
    result = np.asarray(pixels, dtype=np.float32).copy()
    vectors = result[:, :, :3].astype(np.float32) * 2.0 - 1.0
    lengths = np.linalg.norm(vectors, axis=2)
    degenerate = lengths < 1e-6
    safe_lengths = np.where(degenerate, 1.0, lengths)
    vectors = vectors / safe_lengths[:, :, None]
    vectors[degenerate] = (0.0, 0.0, 1.0)
    result[:, :, :3] = np.clip(vectors * 0.5 + 0.5, 0.0, 1.0)
    return result


def normal_bake_report(
    *,
    texture_size: int,
    miss_fraction: float | None,
    target_shading: str,
    baked: bool = True,
    space: str = NORMAL_MAP_SPACE,
    green_channel: str = NORMAL_MAP_GREEN_CHANNEL,
    colorspace: str = NORMAL_MAP_COLORSPACE,
) -> dict:
    """Provenance for the tangent-space normal bake, recorded on the artifact.

    Every field here is one of the traps that silently produces a normal map
    that renders WRONG rather than one that fails loudly: an object-space bake
    exported as tangent-space, a green channel in DirectX convention, an sRGB
    colour space applied to a vector texture, or a flat-shaded target whose
    faceting the bake encodes instead of the source's detail. Writing them
    down is what makes a future "the normal map looks inverted" a lookup
    rather than an investigation.
    """
    if texture_size <= 0:
        raise ValueError(f"normal_bake_report needs a positive texture size, got {texture_size}")
    if target_shading not in {"smooth", "flat"}:
        raise ValueError(
            f"normal_bake_report target_shading must be 'smooth' or 'flat', got {target_shading!r}"
        )
    if miss_fraction is not None and not 0.0 <= float(miss_fraction) <= 1.0:
        raise ValueError(
            f"normal_bake_report miss_fraction must be a fraction, got {miss_fraction!r}"
        )
    return {
        "baked": bool(baked),
        "image_name": bake_image_name("Normal"),
        "texture_size": int(texture_size),
        "space": space,
        "green_channel": green_channel,
        "colorspace": colorspace,
        "target_shading": target_shading,
        "miss_fraction": (
            round(float(miss_fraction), 6) if miss_fraction is not None else None
        ),
    }


def bake_texture_size(
    source_texture_dimensions: Iterable[int],
    *,
    minimum: int = BAKE_TEXTURE_SIZE_MINIMUM,
    maximum: int = BAKE_TEXTURE_SIZE_MAXIMUM,
) -> int:
    """Bake-atlas edge length derived from the source's largest texture.

    `source_texture_dimensions` is every width/height of every image the
    source materials actually sample. The largest is rounded UP to a power of
    two and clamped into [`minimum`, `maximum`]; a source with no textures at
    all falls back to `minimum`. Rounding up never invents detail -- the bake
    resamples the source through Cycles either way -- it only stops the atlas
    from being the resolution bottleneck.
    """
    if minimum <= 0:
        raise ValueError(f"bake_texture_size minimum must be positive, got {minimum}")
    if maximum < minimum:
        raise ValueError(
            f"bake_texture_size maximum ({maximum}) must be >= minimum ({minimum})"
        )
    largest = 0
    for value in source_texture_dimensions:
        number = int(value)
        if number > largest:
            largest = number
    if largest <= 0:
        return minimum
    power_of_two = 1 << (largest - 1).bit_length()
    return max(minimum, min(maximum, power_of_two))


def uv_pack_margin(
    texture_size: int, *, texels: int = UV_PACK_MARGIN_TEXELS
) -> float:
    """`bpy.ops.uv.pack_islands` margin that means `texels` texels at this size.

    The margin is a fraction of the UV unit square, so the same number means
    twice as many texels on a 4K atlas as on a 2K one. Expressing the gutter
    in texels keeps island spacing (and therefore the bake's seam bleed
    budget) constant as the atlas size follows the source.
    """
    if texture_size <= 0:
        raise ValueError(f"uv_pack_margin needs a positive texture size, got {texture_size}")
    if texels < 0:
        raise ValueError(f"uv_pack_margin needs a non-negative texel gutter, got {texels}")
    return min(0.25, texels / float(texture_size))


def unique_by_identity(items: list[object]) -> list[object]:
    """Return first-seen objects without processing shared datablocks twice."""
    seen: set[int] = set()
    unique = []
    for item in items:
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(item)
    return unique


def fill_transparent_texels(
    pixels: np.ndarray,
    *,
    max_passes: int = 64,
) -> np.ndarray:
    """Dilate covered RGBA texels into nearby transparent bake misses."""
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("Texture pixels must be an H x W x 4 RGBA array")
    filled = np.asarray(pixels, dtype=np.float32).copy()
    valid = filled[:, :, 3] >= 0.5
    for _ in range(max_passes):
        missing = ~valid
        if not missing.any():
            break
        color_sum = np.zeros_like(filled)
        neighbor_count = np.zeros(valid.shape, dtype=np.float32)
        for row_offset, column_offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted_valid = np.roll(valid, (row_offset, column_offset), axis=(0, 1))
            shifted_pixels = np.roll(filled, (row_offset, column_offset), axis=(0, 1))
            if row_offset == -1:
                shifted_valid[-1, :] = False
            elif row_offset == 1:
                shifted_valid[0, :] = False
            if column_offset == -1:
                shifted_valid[:, -1] = False
            elif column_offset == 1:
                shifted_valid[:, 0] = False
            color_sum += shifted_pixels * shifted_valid[:, :, None]
            neighbor_count += shifted_valid
        newly_filled = missing & (neighbor_count > 0)
        if not newly_filled.any():
            break
        filled[newly_filled] = (
            color_sum[newly_filled]
            / neighbor_count[newly_filled, None]
        )
        filled[newly_filled, 3] = 1.0
        valid |= newly_filled
    # The reconstructed deformation shell is opaque. Any texels beyond the
    # bounded dilation radius retain their neutral generated RGB fallback, but
    # must not punch transparency holes through the exported GLB material.
    filled[:, :, 3] = 1.0
    return filled


# ---------------------------------------------------------------------------
# Palette fidelity
#
# TRELLIS.2's texture head is generative: when the conditioning image carries
# no palette to copy -- a monochrome, greyscale or heavily stylised source --
# it falls back to its own coloured prior and INVENTS one. Measured 2026-09-01
# on run 62107a15 (a black-and-white punk character): source image abdc6d30
# colorfulness 16.23, the prepared conditioning image 9356577b 8.37, and the
# generated mesh d5fa531d came back with pink skin and blue jeans.
#
# The fix is deterministic and lives in the bake, not in another diffusion
# pass: the canonicalize bake already resamples the source's appearance into a
# fresh atlas, so the atlas is re-statisticked against the conditioning image
# the generator was actually given. A Reinhard mean/std match in Oklab moves
# the albedo's colour distribution onto the reference's without touching which
# texel is which, so it cannot invent, blur or relocate detail -- and when the
# reference has no chroma at all the match drives the albedo's chroma to zero,
# which is exactly the operator's ask.
# ---------------------------------------------------------------------------

# Border ring the reference image's background colour is read from, and the
# keying tolerances applied to it. Same approach as
# `view_consistency.subject_silhouette` (median of a border ring, not a mean
# of four corners, because one shadowed corner drags a mean far enough to key
# half the frame in) -- deliberately REIMPLEMENTED here rather than imported,
# so the bake path stays a pure-numpy module Blender's bundled interpreter can
# load without pulling in PIL and the view-consistency policy surface.
PALETTE_BACKGROUND_RING_PX = 8
# Tolerances are expressed on the 0-255 sum-of-absolute-channel-difference
# scale `view_consistency` calibrated them on, then applied to [0, 1] arrays.
PALETTE_BACKGROUND_KEY_TOLERANCE = 24.0 / 255.0
PALETTE_BACKGROUND_UNIFORMITY_MAX = 24.0 / 255.0
# Below this many keyed pixels the reference has no subject worth taking
# statistics from and the transfer declines to run.
PALETTE_MIN_SUBJECT_PIXELS = 256
# How much of the LUMINANCE match to apply. Chroma is matched fully -- that is
# the whole point -- but L is only nudged: the reference is a SHADED image and
# the albedo is unlit base colour, so matching L fully would bake the
# reference's exposure and its lighting into the texture, and matching its
# standard deviation fully would crush or explode the albedo's own contrast.
PALETTE_LUMINANCE_STRENGTH = 0.35
# Guard rails on the per-channel standard-deviation gain. A near-flat albedo
# channel would otherwise divide by ~0 and blow the texture out.
PALETTE_MIN_STD = 1e-4
PALETTE_MAX_CHROMA_GAIN = 3.0
PALETTE_MAX_LUMINANCE_GAIN = 2.0

# Hue histogram used by the palette DISTANCE metric (not by the transfer).
PALETTE_HUE_BINS = 36
# Oklab chroma below which a pixel has no meaningful hue and must not vote in
# the hue histogram: an achromatic pixel's atan2(b, a) is numerical noise, and
# a mono image is nothing but achromatic pixels.
PALETTE_HUE_MIN_CHROMA = 0.02
# `colorfulness_ratio` is compared in log2 space, so a ratio of 2x and one of
# 0.5x cost the same. This epsilon (in Hasler-Susstrunk 0-255 units) keeps the
# log finite for a genuinely achromatic pair; 1.0 unit is far below the ~8-16
# range a real image occupies.
PALETTE_COLORFULNESS_EPSILON = 1.0
# Where the colorfulness term saturates, in log2 units (2.0 == a 4x mismatch).
PALETTE_COLORFULNESS_SATURATION = 2.0
# Fraction of an image's measured pixels that must carry real chroma (>=
# `PALETTE_HUE_MIN_CHROMA`) before its hue histogram means anything.
#
# Without this gate the hue term is actively misleading on exactly the case
# this module exists for. A monochrome reference has no hues, so its histogram
# is empty; a render of ANY texture has a little chroma from lighting and 8-bit
# quantisation, so its histogram is populated; and chi-square between an empty
# and a populated histogram is near-maximal NO MATTER how well the palette was
# matched. Measured 2026-09-01 on the punk chain: the hue term read 0.74 both
# before and after a correction that visibly worked, while the colorfulness
# term moved 0.357 -> 0.141 as it should.
#
# So: "which hues" is only a question worth asking when both sides HAVE hues.
# When either does not, colorfulness ratio carries the metric alone -- which is
# the right question for a monochrome source anyway ("how much colour is here",
# not "which"). The two chains sit far apart on this measure: the punk
# reference has <1% of subject pixels above the chroma floor, the alien's ~45%.
PALETTE_HUE_EVALUABLE_FRACTION = 0.10

# Oklab (Bjorn Ottosson, 2020). Chosen over CIELAB because it takes LINEAR
# sRGB directly -- no white point to agree on with Blender -- and because its
# a/b axes are perceptually uniform enough that a mean/std match in them does
# not swing hue the way a match in raw RGB does.
_OKLAB_LMS_FROM_LINEAR = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=np.float64)
_OKLAB_FROM_LMS = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
], dtype=np.float64)
_LMS_FROM_OKLAB = np.array([
    [1.0, 0.3963377774, 0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480],
], dtype=np.float64)
_LINEAR_FROM_LMS = np.array([
    [4.0767416621, -3.3077115913, 0.2309699292],
    [-1.2684380046, 2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147, 1.7076147010],
], dtype=np.float64)


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """sRGB-encoded [0, 1] -> linear [0, 1] (the IEC 61966-2-1 transfer curve).

    Blender hands out LINEAR floats from `image.pixels` for an sRGB-tagged
    image, so the bake path never needs this; the palette METRIC does, because
    it reads renders back through PIL, which decodes to sRGB-encoded bytes.
    """
    array = np.asarray(srgb, dtype=np.float64)
    return np.where(array <= 0.04045, array / 12.92, ((array + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Linear [0, 1] -> sRGB-encoded [0, 1]. Inverse of `srgb_to_linear`."""
    array = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    return np.where(array <= 0.0031308, array * 12.92, 1.055 * array ** (1.0 / 2.4) - 0.055)


def linear_rgb_to_oklab(linear_rgb: np.ndarray) -> np.ndarray:
    """Linear sRGB (..., 3) -> Oklab (..., 3) as (L, a, b)."""
    array = np.asarray(linear_rgb, dtype=np.float64)
    if array.shape[-1] != 3:
        raise ValueError(f"linear_rgb_to_oklab needs a trailing RGB axis, got {array.shape}")
    lms = array @ _OKLAB_LMS_FROM_LINEAR.T
    # np.cbrt is sign-safe, which matters: an out-of-gamut or slightly
    # negative linear value must not produce a NaN and poison the statistics.
    return np.cbrt(lms) @ _OKLAB_FROM_LMS.T


def oklab_to_linear_rgb(oklab: np.ndarray) -> np.ndarray:
    """Oklab (..., 3) -> linear sRGB (..., 3), clipped back into gamut."""
    array = np.asarray(oklab, dtype=np.float64)
    if array.shape[-1] != 3:
        raise ValueError(f"oklab_to_linear_rgb needs a trailing Lab axis, got {array.shape}")
    lms = array @ _LMS_FROM_OKLAB.T
    return np.clip((lms ** 3) @ _LINEAR_FROM_LMS.T, 0.0, 1.0)


def estimate_background_color(rgb: np.ndarray) -> np.ndarray | None:
    """Median colour of a uniform border ring, or None when the ring is not
    uniform enough to be trusted as a background.

    Split out of `subject_pixel_mask` so a caller that already has a subject
    mask (and just wants to re-check one specific colour, e.g. a bilinear-
    sampled projection colour, against the same background estimate) does
    not have to re-derive it by hand. Median of a ring, not a mean of four
    corners: see `subject_pixel_mask`'s docstring for why.
    """
    array = np.asarray(rgb, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"estimate_background_color needs an H x W x 3 array, got {array.shape}")
    ring_px = min(PALETTE_BACKGROUND_RING_PX, max(1, min(array.shape[0], array.shape[1]) // 8))
    ring = np.concatenate([
        array[:ring_px].reshape(-1, 3),
        array[-ring_px:].reshape(-1, 3),
        array[:, :ring_px].reshape(-1, 3),
        array[:, -ring_px:].reshape(-1, 3),
    ])
    background = np.median(ring, axis=0)
    dispersion = float(np.percentile(np.abs(ring - background).sum(axis=1), 95))
    if dispersion > PALETTE_BACKGROUND_UNIFORMITY_MAX:
        return None
    return background


def subject_pixel_mask(rgb: np.ndarray, alpha: np.ndarray | None = None) -> np.ndarray | None:
    """Boolean subject mask for a reference image, or None if unkeyable.

    Uses real transparency when the image carries it. Otherwise it keys
    against the MEDIAN colour of a border ring (`estimate_background_color`)
    and refuses to key at all when that ring is not uniform enough to be a
    background -- a lighting gradient or a cast floor shadow would otherwise
    be measured as subject, and palette statistics taken over a shadow are
    worse than no transfer.

    Refusing (returning None) is a first-class outcome here: the caller falls
    back to the whole frame or skips the transfer, rather than shifting a
    texture's entire colour distribution toward a backdrop.
    """
    array = np.asarray(rgb, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"subject_pixel_mask needs an H x W x 3 array, got {array.shape}")
    if alpha is not None:
        alpha_array = np.asarray(alpha, dtype=np.float64)
        if alpha_array.shape != array.shape[:2]:
            raise ValueError(
                f"subject_pixel_mask alpha shape {alpha_array.shape} does not match image {array.shape[:2]}"
            )
        if float(alpha_array.min()) < 0.98:
            return alpha_array > 0.5
    background = estimate_background_color(array)
    if background is None:
        return None
    return np.abs(array - background).sum(axis=2) > PALETTE_BACKGROUND_KEY_TOLERANCE


def _selected(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Flatten an (H, W, C) array to (N, C) over `mask` (or all pixels)."""
    if mask is None:
        return values.reshape(-1, values.shape[-1])
    selection = np.asarray(mask, dtype=bool)
    if selection.shape != values.shape[:2]:
        raise ValueError(
            f"palette mask shape {selection.shape} does not match image {values.shape[:2]}"
        )
    return values[selection]


def palette_statistics(
    linear_rgb: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, Any]:
    """Oklab mean/std plus colorfulness over `mask`, for provenance and tests.

    `linear_rgb` is an H x W x 3 LINEAR array; `colorfulness` is reported in
    the Hasler-Susstrunk 0-255 units the operator's numbers are quoted in
    (see `image_colorfulness`).
    """
    array = np.asarray(linear_rgb, dtype=np.float64)
    lab = _selected(linear_rgb_to_oklab(array), mask)
    if lab.shape[0] == 0:
        return {
            "pixels": 0,
            "mean": [0.0, 0.0, 0.0],
            "std": [0.0, 0.0, 0.0],
            "colorfulness": 0.0,
        }
    return {
        "pixels": int(lab.shape[0]),
        "mean": [round(float(value), 6) for value in lab.mean(axis=0)],
        "std": [round(float(value), 6) for value in lab.std(axis=0)],
        "colorfulness": round(image_colorfulness(linear_to_srgb(array), mask), 4),
    }


def image_colorfulness(srgb: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Hasler-Susstrunk colorfulness of an sRGB-ENCODED [0, 1] image, 0-255 scale.

    The published metric (Hasler & Susstrunk 2003) on 8-bit channels, so the
    numbers here are directly comparable to the ones quoted against the
    operator's artifacts: source image abdc6d30 scores 16.23, the prepared
    conditioning image 9356577b 8.37. Roughly: below ~15 an image is
    near-monochrome, above ~40 it is vividly coloured.

    Note the input space. Colorfulness is a PERCEPTUAL quantity and is defined
    on display-encoded values; computing it on linear floats reads far lower
    and would not match any published figure.
    """
    array = np.asarray(srgb, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"image_colorfulness needs an H x W x 3 array, got {array.shape}")
    pixels = _selected(array, mask) * 255.0
    if pixels.shape[0] == 0:
        return 0.0
    red, green, blue = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    rg = red - green
    yb = 0.5 * (red + green) - blue
    std = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2))
    mean = float(np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return std + 0.3 * mean


def hue_histogram(
    srgb: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    bins: int = PALETTE_HUE_BINS,
    min_chroma: float = PALETTE_HUE_MIN_CHROMA,
) -> np.ndarray:
    """Chroma-weighted Oklab hue histogram over `mask`, summing to 1.0.

    Weighted by chroma and cut below `min_chroma` so that a monochrome image
    produces an EMPTY (all-zero) histogram rather than a uniform smear of
    numerical noise: the hue of a grey pixel is meaningless, and letting grey
    pixels vote would make every mono image look hue-identical to every other.
    An all-zero histogram is the honest answer for "this image has no hues",
    and `palette_distance` handles that case explicitly.
    """
    if bins <= 0:
        raise ValueError(f"hue_histogram needs a positive bin count, got {bins}")
    array = np.asarray(srgb, dtype=np.float64)
    lab = _selected(linear_rgb_to_oklab(srgb_to_linear(array)), mask)
    if lab.shape[0] == 0:
        return np.zeros(bins, dtype=np.float64)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    keep = chroma >= min_chroma
    if not keep.any():
        return np.zeros(bins, dtype=np.float64)
    hue = np.arctan2(lab[keep, 2], lab[keep, 1])
    index = np.clip(((hue + np.pi) / (2.0 * np.pi) * bins).astype(int), 0, bins - 1)
    histogram = np.bincount(index, weights=chroma[keep], minlength=bins).astype(np.float64)
    total = float(histogram.sum())
    return histogram / total if total > 0.0 else histogram


def chromatic_fraction(
    srgb: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    min_chroma: float = PALETTE_HUE_MIN_CHROMA,
) -> float:
    """Fraction of measured pixels carrying real chroma -- "does this have hues".

    The gate on whether a hue comparison means anything at all (see
    `PALETTE_HUE_EVALUABLE_FRACTION`).
    """
    lab = _selected(linear_rgb_to_oklab(srgb_to_linear(np.asarray(srgb, dtype=np.float64))), mask)
    if lab.shape[0] == 0:
        return 0.0
    return float(np.mean(np.hypot(lab[:, 1], lab[:, 2]) >= min_chroma))


def _chi_square(first: np.ndarray, second: np.ndarray) -> float:
    """Symmetric chi-square distance between two normalized histograms, [0, 1].

    Two histograms that are both empty (both images achromatic) are IDENTICAL
    -- distance 0 -- and one empty against one populated is maximally
    different. Getting this pair right is the whole mono case: the pre-fix
    artifact has hues the reference does not, and that must read as far apart.
    """
    numerator = (first - second) ** 2
    denominator = first + second
    nonzero = denominator > 0.0
    if not nonzero.any():
        return 0.0
    return float(0.5 * np.sum(numerator[nonzero] / denominator[nonzero]))


def palette_distance(
    reference_srgb: np.ndarray,
    candidate_srgb: np.ndarray,
    *,
    reference_mask: np.ndarray | None = None,
    candidate_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """How far a candidate's palette has drifted from its reference's, in [0, 1].

    Both inputs are sRGB-ENCODED H x W x 3 arrays in [0, 1]; they need not
    share a shape, because this compares DISTRIBUTIONS (a 768px render against
    a 1024px conditioning image) rather than pixels.

    Two independent signals, averaged:

    * `colorfulness_term` -- the log2 ratio of Hasler-Susstrunk colorfulness,
      saturating at a 4x mismatch. This is what catches the operator's case:
      a mono reference against an invented-palette candidate.
    * `hue_term` -- symmetric chi-square between chroma-weighted hue
      histograms. This catches a candidate that is as colourful as its
      reference but coloured WRONG (correct saturation, blue where the source
      is red), which the ratio alone is blind to. It is only INCLUDED when
      both sides actually carry hues (`PALETTE_HUE_EVALUABLE_FRACTION`); for a
      monochrome reference the question is meaningless and the colorfulness
      ratio carries the metric alone. `hue_evaluable` records which happened.

    Deliberately asymmetric in interpretation but symmetric in arithmetic: a
    candidate that is too drab scores the same as one that is too vivid,
    because both are the texture head failing to reproduce the conditioning
    image's palette.
    """
    reference_colorfulness = image_colorfulness(reference_srgb, reference_mask)
    candidate_colorfulness = image_colorfulness(candidate_srgb, candidate_mask)
    ratio = (
        (candidate_colorfulness + PALETTE_COLORFULNESS_EPSILON)
        / (reference_colorfulness + PALETTE_COLORFULNESS_EPSILON)
    )
    colorfulness_term = min(
        1.0, abs(float(np.log2(ratio))) / PALETTE_COLORFULNESS_SATURATION
    )
    hue_term = _chi_square(
        hue_histogram(reference_srgb, reference_mask),
        hue_histogram(candidate_srgb, candidate_mask),
    )
    hue_evaluable = (
        min(
            chromatic_fraction(reference_srgb, reference_mask),
            chromatic_fraction(candidate_srgb, candidate_mask),
        )
        >= PALETTE_HUE_EVALUABLE_FRACTION
    )
    distance = (
        0.5 * colorfulness_term + 0.5 * hue_term if hue_evaluable else colorfulness_term
    )
    return {
        "colorfulness_reference": round(float(reference_colorfulness), 4),
        "colorfulness_candidate": round(float(candidate_colorfulness), 4),
        "colorfulness_ratio": round(float(ratio), 4),
        "colorfulness_term": round(float(colorfulness_term), 6),
        "hue_term": round(float(hue_term), 6),
        "hue_evaluable": bool(hue_evaluable),
        "distance": round(float(distance), 6),
    }


def transfer_palette(
    albedo_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    reference_mask: np.ndarray | None = None,
    strength: float = 1.0,
    *,
    albedo_mask: np.ndarray | None = None,
    luminance_strength: float = PALETTE_LUMINANCE_STRENGTH,
) -> np.ndarray:
    """Reinhard mean/std colour transfer from a reference image onto an albedo.

    Both arrays are H x W x 3 LINEAR RGB (what Blender's `image.pixels` hands
    out for an sRGB-tagged image, in both directions -- no encode/decode round
    trip happens inside the bake). They need not share a shape.

    `reference_mask` selects the reference's SUBJECT pixels, so the transfer
    matches the character rather than the backdrop; `albedo_mask` selects the
    bake's covered texels, so the uncovered atlas -- neutral fallback grey and
    gutter -- is neither measured nor rewritten. Both default to "every pixel".

    The match runs in Oklab and treats the axes differently:

    * `a` and `b` (chroma) are matched FULLY, mean and standard deviation.
      This is the entire fix. When the reference is monochrome its chroma mean
      and standard deviation are both ~0, so the output's chroma is driven to
      ~0 too and the texture comes out grey -- no special case, just the
      arithmetic doing what it says.
    * `L` (luminance) is matched only `luminance_strength` of the way. The
      reference is a shaded image and the albedo is unlit base colour, so a
      full match would bake the reference's exposure and lighting into the
      texture and crush the albedo's own contrast.

    `strength` linearly blends the result back toward the input, so
    `strength=0.0` is an exact no-op and `strength=1.0` is the full match.

    Texels outside `albedo_mask` are returned byte-identical. Nothing here
    moves a texel, so it cannot blur, shift or invent detail: every pixel's
    new value depends only on its own old value and two sets of scalars.
    """
    albedo = np.asarray(albedo_rgb, dtype=np.float64)
    reference = np.asarray(reference_rgb, dtype=np.float64)
    if albedo.ndim != 3 or albedo.shape[2] != 3:
        raise ValueError(f"transfer_palette needs an H x W x 3 albedo, got {albedo.shape}")
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError(f"transfer_palette needs an H x W x 3 reference, got {reference.shape}")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError(f"transfer_palette strength must be in [0, 1], got {strength}")
    if not 0.0 <= float(luminance_strength) <= 1.0:
        raise ValueError(
            f"transfer_palette luminance_strength must be in [0, 1], got {luminance_strength}"
        )
    if float(strength) == 0.0:
        return albedo.copy()

    reference_lab = _selected(linear_rgb_to_oklab(reference), reference_mask)
    if reference_lab.shape[0] == 0:
        return albedo.copy()
    albedo_lab = linear_rgb_to_oklab(albedo)
    selected_albedo = _selected(albedo_lab, albedo_mask)
    if selected_albedo.shape[0] == 0:
        return albedo.copy()

    reference_mean = reference_lab.mean(axis=0)
    reference_std = reference_lab.std(axis=0)
    albedo_mean = selected_albedo.mean(axis=0)
    albedo_std = selected_albedo.std(axis=0)

    # Per-channel standard-deviation gain. A near-flat albedo channel gets a
    # gain of exactly 1.0 rather than a division by ~0: there is no spread to
    # rescale, and the mean shift alone already places it.
    gain = np.where(
        albedo_std > PALETTE_MIN_STD, reference_std / np.maximum(albedo_std, PALETTE_MIN_STD), 1.0
    )
    gain[0] = 1.0 + luminance_strength * (gain[0] - 1.0)
    gain[0] = float(np.clip(gain[0], 1.0 / PALETTE_MAX_LUMINANCE_GAIN, PALETTE_MAX_LUMINANCE_GAIN))
    gain[1:] = np.clip(gain[1:], 0.0, PALETTE_MAX_CHROMA_GAIN)

    # Target mean. Chroma moves all the way onto the reference's; luminance
    # only `luminance_strength` of the way.
    target_mean = reference_mean.copy()
    target_mean[0] = albedo_mean[0] + luminance_strength * (reference_mean[0] - albedo_mean[0])

    matched = (albedo_lab - albedo_mean) * gain + target_mean
    blended = albedo_lab + float(strength) * (matched - albedo_lab)

    result = albedo.copy()
    write = (
        np.ones(albedo.shape[:2], dtype=bool)
        if albedo_mask is None
        else np.asarray(albedo_mask, dtype=bool)
    )
    if write.shape != albedo.shape[:2]:
        raise ValueError(
            f"transfer_palette albedo_mask shape {write.shape} does not match albedo {albedo.shape[:2]}"
        )
    result[write] = oklab_to_linear_rgb(blended[write])
    return result


def palette_transfer_report(
    *,
    applied: bool,
    reason: str | None = None,
    reference_id: str | None = None,
    strength: float | None = None,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    reference: dict[str, Any] | None = None,
    distance_before: dict[str, Any] | None = None,
    distance_after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Provenance for one palette transfer, recorded on the artifact.

    `applied=False` with a `reason` is a first-class result, not an error: a
    canonicalize whose lineage has no conditioning image, or whose reference
    has no keyable subject, must still produce an artifact and must still say
    in its metadata WHY the texture was left alone.

    `distance_before`/`distance_after` are `palette_distance` results measured
    on the ATLAS against the reference. Keeping BEFORE is what stops the
    metric from being a tautology once the transfer is on: `after` says the
    shipped texture matches the source, `before` says how far the generator
    had drifted on its own -- the number that would otherwise vanish the
    moment the fix started working.
    """
    return {
        "applied": bool(applied),
        "reason": reason,
        "reference_artifact_id": reference_id,
        "strength": round(float(strength), 4) if strength is not None else None,
        "reference_colorfulness": (reference or {}).get("colorfulness"),
        "albedo_colorfulness_before": (before or {}).get("colorfulness"),
        "albedo_colorfulness_after": (after or {}).get("colorfulness"),
        "palette_distance_before": distance_before,
        "palette_distance_after": distance_after,
        "reference_statistics": reference,
        "albedo_statistics_before": before,
        "albedo_statistics_after": after,
    }
