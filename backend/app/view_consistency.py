"""Does a synthesized view CONTRADICT the front reference it was drawn from?

View synthesis redraws a character from a new camera angle. When it works,
only the camera moved. When it fails, the model reinterprets the
*character* -- and the failure that motivated this module is the worst kind,
because every existing gate passed it:

    The prepared front is a fitted purple jumpsuit with two clearly
    separated legs and boots. The synthesized LEFT view is a floor-length
    robe: one fused, skirt-shaped lower body, no leg separation at all.
    TRELLIS.2 was then conditioned on both and reconciled them into a
    fused-leg skirt mesh that no amount of downstream rigging can fix.

    The multi-view bbox-height check in `trellis2_fp8_worker` (the 18.3%
    warning) did NOT catch it: the robe and the jumpsuit are the same
    HEIGHT. Height agreement says nothing about garment silhouette.

So this module measures the subject's *lower-body silhouette shape* and asks
whether the view is still depicting the same garment as the front. Pure
numpy + Pillow, no GPU, no new dependencies, no model calls -- it runs in
well under a second per pair, cheaply enough to sit in the synthesize-views
job and in any pre-generate check.

## The signals

Everything is measured on the subject silhouette (real alpha when the image
has any, otherwise keyed against a background colour sampled from the four
corners -- every profile in `image_preparation` demands a plain neutral
background, so corner sampling is a sound background estimate here) inside
the LOWER BAND: the bottom 40% of the subject's robust vertical extent,
which is hips-to-floor on a standing humanoid.

(a) `gap_row_fraction` -- the fraction of lower-band rows that contain a
    background gap strictly between the silhouette's own left and right
    edge, wide enough (>= 1.5% of subject scale) not to be antialiasing.
    Two separated legs punch a gap through nearly every row of the band;
    a skirt, robe, or gown punches none.

(b) `lower_width_ratio` -- the view's median lower-band silhouette width
    over the front's, each normalised by its own subject scale. This is the
    signal that makes (a) usable on a PROFILE view, see "Why a side view
    needs both" below.

(c) `lower_profile_corr` -- Pearson correlation of the two per-row width
    profiles over the band, resampled to a fixed bin count. Discriminating
    on front-plane views (a fused or reshaped lower body decorrelates the
    row profile); NOT discriminating on profile views, and deliberately not
    gated there -- see "Calibration" below.

(d) `lower_color_distance` -- total-variation distance between coarse 4x4x4
    RGB histograms of the lower band. RECORDED, NEVER GATED: the motivating
    failure's robe is the same purple as its jumpsuit, so colour agreement
    proves nothing, and colour disagreement has legitimate causes (a side
    view is lit differently). Silhouette is the primary signal.

## Why a side view needs both (a) and (b)

A 90-degree profile view of a genuinely two-legged character shows NO gap
between the legs -- they occlude each other. Gap-loss alone would therefore
condemn every good left/right view ever synthesized (measured: a known-good
left view in the calibration set scores gap_row_fraction=0.003 against a
front that scores 1.000). What separates a good profile view from a robe is
the WIDTH: seen edge-on, a pair of separated legs collapses to roughly one
leg's depth and gets NARROWER than the front's leg span, while a robe hem
stays as wide or wider. So on a profile view the check requires both a lost
gap AND a lower body that failed to narrow.

## The skirted-character case (deliberate non-firing)

A character who genuinely wears a floor-length skirt will show no leg gap in
ANY view -- including the front. Because every rule here is preconditioned on
`front_gap_row_fraction >= FRONT_LEG_SEPARATION_MIN`, i.e. on the FRONT
itself demonstrating separated legs, a consistently-skirted character can
never trip `leg_separation_mismatch`: there is no separation in the reference
for the view to contradict. This was verified, not asserted, on the
calibration set: the original source art behind the motivating failure is
itself a robed figure (front gap_row_fraction=0.018); all three views
synthesized from it measure clean against it, and the SAME left view
measures dirty against the prepared jumpsuit front. The CC0 fixtures
reproduce this (second table below). The check compares against the front,
so it flags the disagreement and takes no view of which image is "right".

## Robust normalisation and refusing to score

Two failure modes seen in earlier silhouette-matching work are designed out
here rather than retrofitted:

* Scale normalisation is by a ROBUST PERCENTILE extent, not by the raw
  min/max bounding box. `_robust_extent` trims the outermost
  `SCALE_TRIM_FRACTION` of silhouette AREA from each end, so a drop shadow,
  a stray hair wisp, or one antialiased speck cannot move the denominator
  that every width is divided by. Min/max bbox normalisation is exactly what
  produced incoherent cross-view readings there.
* No alignment or offset SEARCH is performed at all. The band is anchored
  deterministically to each image's own robust extent and both profiles are
  resampled to a fixed bin count, so there is no optimisation that could
  silently lock to its search bounds and corrupt a score. If a future
  revision adds a search, it must assert the optimum is interior to the
  bounds and report `not_evaluable` when it is not.

And when the inputs do not support a measurement, this module REFUSES to
score rather than degrading to a weaker comparison: a failed background key
(silhouette coverage outside `COVERAGE_MIN..COVERAGE_MAX`), an empty mask, a
band too short to resample, or a degenerate constant profile all yield a
single `not-evaluable` finding and no verdict. Every result publishes its
denominators -- `front_band_rows`, `view_band_rows`, `profile_bins`, and both
coverages -- alongside the scores, so a reader can always see how much
silhouette the numbers were actually computed over.

## Calibration

The thresholds were calibrated on an internal set of synthesized views, each
measured against the front it was synthesized from: 40 known-good
view/front pairs plus the two drifted views of the motivating failure (its
left and back views, both redrawn as a robe against the jumpsuit front).
Across the 40 known-good pairs the widest `lower_width_ratio` is 1.12 and
the lowest front-plane `lower_profile_corr` is 0.897; the two drifted views
measure 1.46/2.00 (ratio, left/back) and 0.61/0.75 (corr, left/back). The
thresholds below sit in those gaps.

The committed test fixtures (`backend/tests/fixtures/view_consistency/`,
Blender renders of CC0 models; see PROVENANCE.md there) pin the same
behaviour and measure:

  front -> view                          view  front_gap  view_gap  ratio    corr  status
  front-legs -> left-robe-drift          left      1.000     0.000  2.462  -0.763  ERROR
  front-legs -> back-robe-drift          back      1.000     0.000  2.762  -0.763  ERROR
  front-zombie -> left-zombie-good       left      0.720     0.000  0.548   0.615  clean
  front-zombie -> back-zombie-good       back      0.720     0.720  1.000   1.000  clean
  front-robe-source -> left-robe-drift   left      0.000     0.000  0.891   0.994  clean (no
                                                            separation in the front to contradict)

The renders are cleaner than synthesized views (orthographic, symmetric,
flat lighting on a plain background), so they sit further from every
threshold than the internal drifted views did: they pin the direction of
each rule, not its margin.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Literal

CHECK_VERSION = 2

Severity = Literal["warning", "error", "not-evaluable"]

LEG_SEPARATION_MISMATCH = "view_consistency.leg_separation_mismatch"
LOWER_SILHOUETTE_DECORRELATED = "view_consistency.lower_silhouette_decorrelated"
VALUE_COLLAPSE = "view_consistency.value_collapse"
NOT_EVALUABLE = "view_consistency.not_evaluable"

# Views that share the front's camera plane: leg separation visible in the
# front MUST still be visible here, because no foreshortening can hide it.
FRONT_PLANE_VIEWS = frozenset({"front", "back"})
# Views 90 degrees off: legs occlude each other, so gap loss is expected and
# only a lower body that also failed to NARROW is evidence of a garment swap.
PROFILE_VIEWS = frozenset({"left", "right"})

# Silhouette geometry.
BACKGROUND_KEY_TOLERANCE = 24  # matches image_preparation.detect_subject_bbox
BACKGROUND_RING_PX = 8         # border ring the background colour is read from
LOWER_BAND_FRACTION = 0.40     # bottom 40% of the subject: hips to floor
SCALE_TRIM_FRACTION = 0.0025   # robust-percentile trim, each end, by area
MIN_GAP_FRACTION = 0.015       # a "gap" must span >=1.5% of subject scale
PROFILE_BINS = 64
COLOR_LEVELS = 4               # 4x4x4 = 64-bin coarse RGB histogram

# Refuse-to-score guards.
COVERAGE_MIN = 0.01            # nothing keyed as subject
COVERAGE_MAX = 0.85            # background key failed; the frame IS the subject
MIN_BAND_ROWS = 40             # too few rows to resample honestly
# 95th-percentile deviation of the border ring from its own median. Measured
# over the internal calibration images: images with a genuinely plain
# background score <= 16, images carrying a lighting gradient or a cast floor
# shadow score >= 48. Above this the background is not keyable and the
# measurement is refused rather than computed on a shadow.
BACKGROUND_UNIFORMITY_MAX = 24

# Thresholds. See "Calibration" in the module docstring for the measured
# distributions these sit between.
FRONT_LEG_SEPARATION_MIN = 0.35   # front must itself show separated legs
VIEW_LEG_SEPARATION_MIN = 0.35    # below this the view has lost the separation
LOWER_WIDTH_RATIO_ERROR = 1.25    # good <= 1.12, drifted >= 1.46
LOWER_WIDTH_RATIO_WARNING = 1.05  # profile views only; good <= 1.01
FRONT_PLANE_CORR_WARNING = 0.85   # front-plane only; good >= 0.897, drift 0.748
# A view whose keyed subject is dramatically darker than its front reference
# AND carries no bright content anywhere is a backlit / silhouetted render,
# not a lit redraw -- shape checks are structurally blind to this (a solid
# black silhouette matches its front's outline PERFECTLY). BOTH conditions
# are required because the mean ratio alone cannot separate a silhouette
# from a legitimately dark-clothed character: in the internal calibration
# set, a dark-coated character's known-good left profile measures ratio 0.53
# (the coat dominates the profile) yet its p90 luminance is 97 -- real
# highlights exist -- while failed silhouette renders measure p90 around 3
# (one of them unkeyable). Calibration: good views p90 >= 97 (the dark-coated
# profile), 116 (a drifted robe view), 168 (a jumpsuit front); silhouettes
# ~3. The p90 floor sits between with wide margin.
SUBJECT_LUMINANCE_RATIO_MIN = 0.55
SUBJECT_LUMINANCE_P90_FLOOR = 75.0


def finding(
    code: str,
    severity: Severity,
    message: str,
    *,
    metric: str,
    actual: Any,
    expected: str,
) -> dict[str, Any]:
    """Build a finding dict. Shape mirrors `pipeline_policy.finding` and
    `quality_policy._finding`; duplicated rather than imported for the same
    reason `pipeline_policy` duplicates it -- these policy surfaces evolve
    independently and should not couple."""
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "metric": metric,
        "actual": actual,
        "expected": expected,
    }


@dataclass(frozen=True)
class SilhouetteMeasurement:
    """One image's lower-body silhouette signals, all scale-normalised."""

    subject_scale: float
    coverage: float
    band_rows: int
    gap_row_fraction: float
    max_gap_norm: float
    lower_width_median_norm: float
    lower_width_p90_norm: float
    subject_luminance: float
    subject_luminance_p90: float
    width_profile: tuple[float, ...]
    color_histogram: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject_scale": round(self.subject_scale, 2),
            "coverage": round(self.coverage, 4),
            "band_rows": self.band_rows,
            "gap_row_fraction": round(self.gap_row_fraction, 4),
            "max_gap_norm": round(self.max_gap_norm, 4),
            "lower_width_median_norm": round(self.lower_width_median_norm, 4),
            "lower_width_p90_norm": round(self.lower_width_p90_norm, 4),
            "subject_luminance": round(self.subject_luminance, 2),
            "subject_luminance_p90": round(self.subject_luminance_p90, 2),
        }


def subject_silhouette(image_bytes: bytes) -> tuple[Any, Any]:
    """Return (boolean subject mask or None, RGB array) for a reference image.

    Uses the alpha channel when the image carries real transparency. Otherwise
    it keys against the MEDIAN colour of an 8px border ring -- a percentile
    estimate rather than `image_preparation.detect_subject_bbox`'s mean of the
    four corner pixels, because one corner darkened by a cast shadow drags a
    mean far enough to key half the frame in (measured on a calibration image
    with one shadowed corner: the corner mean keys 98.6% of the frame as
    "subject", the ring median 21.4%).

    The mask is None -- refusing to measure -- when the ring is not uniform
    enough to be a background at all: a lighting gradient or a floor shadow
    would otherwise be silhouette, and a silhouette measured on a shadow is
    worse than no measurement."""
    import numpy as np
    from PIL import Image

    with Image.open(BytesIO(image_bytes)) as opened:
        rgba = opened.convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A"))
        rgb = np.asarray(rgba.convert("RGB"), dtype=np.int16)
    if int(alpha.min()) < 250:
        return alpha > 127, rgb
    ring_px = min(BACKGROUND_RING_PX, max(1, min(rgb.shape[0], rgb.shape[1]) // 8))
    ring = np.concatenate([
        rgb[:ring_px].reshape(-1, 3),
        rgb[-ring_px:].reshape(-1, 3),
        rgb[:, :ring_px].reshape(-1, 3),
        rgb[:, -ring_px:].reshape(-1, 3),
    ])
    background = np.median(ring, axis=0)
    dispersion = float(np.percentile(np.abs(ring - background).sum(axis=1), 95))
    if dispersion > BACKGROUND_UNIFORMITY_MAX:
        return None, rgb
    return np.abs(rgb - background).sum(axis=2) > BACKGROUND_KEY_TOLERANCE, rgb


def _robust_extent(row_area: Any) -> tuple[int, int]:
    """First and last row bounding the central `1 - 2*SCALE_TRIM_FRACTION`
    of silhouette AREA.

    This is the robust-percentile alternative to a min/max bounding box: a
    drop shadow, a stray speck, or one antialiased hair strand carries
    almost no area and cannot move these bounds, whereas it moves a raw
    bbox edge arbitrarily far. Every width in this module is divided by the
    extent it defines, so the denominator has to be the stable one."""
    import numpy as np

    total = float(row_area.sum())
    cumulative = np.cumsum(row_area)
    low = int(np.searchsorted(cumulative, total * SCALE_TRIM_FRACTION, side="left"))
    high = int(np.searchsorted(cumulative, total * (1.0 - SCALE_TRIM_FRACTION), side="left"))
    return low, min(high, len(row_area) - 1)


def _interior_gap_run(row: Any) -> int:
    """Longest background run strictly between a row's own outer silhouette
    edges, in pixels. Zero for a row with no silhouette."""
    import numpy as np

    filled = np.flatnonzero(row)
    if filled.size == 0:
        return 0
    interior = row[filled[0]: filled[-1] + 1]
    if interior.all():
        return 0
    # Longest run of False: differences between consecutive True positions.
    true_positions = np.flatnonzero(interior)
    gaps = np.diff(true_positions) - 1
    return int(gaps.max()) if gaps.size else 0


def measure_silhouette(image_bytes: bytes) -> SilhouetteMeasurement | None:
    """Measure one image's lower-body silhouette, or None when the image does
    not support an honest measurement (empty mask, failed background key, or
    a band too short to resample)."""
    import numpy as np

    mask, rgb = subject_silhouette(image_bytes)
    if mask is None:
        return None
    coverage = float(mask.sum()) / float(mask.size)
    if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
        return None
    row_area = mask.sum(axis=1)
    if row_area.sum() <= 0:
        return None
    top, bottom = _robust_extent(row_area)
    scale = float(bottom - top + 1)
    if scale < MIN_BAND_ROWS / LOWER_BAND_FRACTION:
        return None
    band_top = top + int(round(scale * (1.0 - LOWER_BAND_FRACTION)))
    band_mask = mask[band_top: bottom + 1]
    band_rows = int(band_mask.shape[0])
    if band_rows < MIN_BAND_ROWS:
        return None

    columns = np.flatnonzero(mask.any(axis=0))
    band = band_mask[:, columns.min(): columns.max() + 1]
    widths = band.sum(axis=1).astype(float)

    min_gap = max(2, int(round(scale * MIN_GAP_FRACTION)))
    runs = np.array([_interior_gap_run(row) for row in band], dtype=float)
    gap_row_fraction = float((runs >= min_gap).sum()) / band_rows
    max_gap_norm = float(runs.max()) / scale if runs.size else 0.0

    resampled = np.interp(
        np.linspace(0.0, 1.0, PROFILE_BINS),
        np.linspace(0.0, 1.0, band_rows),
        widths,
    ) / scale

    band_rgb = rgb[band_top: bottom + 1, columns.min(): columns.max() + 1]
    quantised = (band_rgb[band] // (256 // COLOR_LEVELS)).astype(int)
    quantised = np.clip(quantised, 0, COLOR_LEVELS - 1)
    histogram = np.zeros(COLOR_LEVELS ** 3, dtype=float)
    if quantised.size:
        flat = (
            quantised[:, 0] * COLOR_LEVELS ** 2
            + quantised[:, 1] * COLOR_LEVELS
            + quantised[:, 2]
        )
        histogram = np.bincount(flat, minlength=COLOR_LEVELS ** 3).astype(float)
        histogram /= histogram.sum()

    subject_pixels = rgb[mask].astype(float)
    if subject_pixels.size:
        per_pixel = subject_pixels @ np.array([0.299, 0.587, 0.114])
        luminance = float(per_pixel.mean())
        luminance_p90 = float(np.percentile(per_pixel, 90))
    else:
        luminance = 0.0
        luminance_p90 = 0.0

    return SilhouetteMeasurement(
        subject_scale=scale,
        coverage=coverage,
        band_rows=band_rows,
        gap_row_fraction=gap_row_fraction,
        max_gap_norm=max_gap_norm,
        lower_width_median_norm=float(np.median(widths)) / scale,
        lower_width_p90_norm=float(np.percentile(widths, 90)) / scale,
        subject_luminance=luminance,
        subject_luminance_p90=luminance_p90,
        width_profile=tuple(float(value) for value in resampled),
        color_histogram=tuple(float(value) for value in histogram),
    )


def view_plane(view: str | None) -> str:
    """Classify a view label into the camera plane its geometry implies.

    Anything unrecognised is treated as `"profile"`, the conservative branch:
    it demands the extra width evidence before calling a mismatch, so an
    unlabelled view can never be condemned on gap loss alone."""
    label = (view or "").strip().lower()
    if label in FRONT_PLANE_VIEWS:
        return "front-plane"
    if label in PROFILE_VIEWS:
        return "profile"
    return "unknown"


def _correlation(first: Any, second: Any) -> float | None:
    import numpy as np

    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-9:
        return None
    return float(a @ b / denominator)


def _not_evaluable(view: str | None, reason: str) -> dict[str, Any]:
    return {
        "check_version": CHECK_VERSION,
        "view": view,
        "view_plane": view_plane(view),
        "evaluable": False,
        "status": "not_evaluable",
        "measurements": {},
        "findings": [
            finding(
                NOT_EVALUABLE,
                "not-evaluable",
                f"View consistency could not be measured: {reason}.",
                metric="silhouette",
                actual=reason,
                expected="a keyable subject silhouette in both images",
            )
        ],
    }


def compare_view_to_front(
    front_bytes: bytes,
    view_bytes: bytes,
    *,
    view: str | None = None,
) -> dict[str, Any]:
    """Does `view_bytes` contradict the front reference `front_bytes`?

    Returns a result dict carrying `measurements` (every signal plus the
    denominators they were computed over), `findings` (the shared
    code/severity/message/metric/actual/expected shape), and a rolled-up
    `status` of `"clean"`, `"warning"`, `"blocked"`, or `"not_evaluable"`.

    Pure: no I/O, no global state, no model calls. Safe to call from a job,
    from a pre-generate gate, or from a test."""
    plane = view_plane(view)
    front = measure_silhouette(front_bytes)
    if front is None:
        return _not_evaluable(
            view,
            "the front reference has no keyable subject silhouette (non-uniform "
            "background, empty or frame-filling mask, or too short a subject)",
        )
    measured = measure_silhouette(view_bytes)
    if measured is None:
        return _not_evaluable(
            view,
            "the synthesized view has no keyable subject silhouette (non-uniform "
            "background, empty or frame-filling mask, or too short a subject)",
        )

    correlation = _correlation(front.width_profile, measured.width_profile)
    if correlation is None:
        return _not_evaluable(view, "a lower-body width profile is constant, so correlation is undefined")

    import numpy as np

    width_ratio = measured.lower_width_median_norm / max(1e-6, front.lower_width_median_norm)
    color_distance = float(
        np.abs(np.asarray(front.color_histogram) - np.asarray(measured.color_histogram)).sum() / 2.0
    )
    luminance_ratio = measured.subject_luminance / max(1e-6, front.subject_luminance)

    measurements = {
        "front": front.as_dict(),
        "view": measured.as_dict(),
        "lower_width_ratio": round(width_ratio, 4),
        "lower_profile_corr": round(correlation, 4),
        "lower_color_distance": round(color_distance, 4),
        "subject_luminance_ratio": round(luminance_ratio, 4),
        # Denominators: how much silhouette every score above was computed
        # over. Published with the scores on purpose -- a number without its
        # denominator cannot be argued with.
        "front_band_rows": front.band_rows,
        "view_band_rows": measured.band_rows,
        "profile_bins": PROFILE_BINS,
        "front_coverage": round(front.coverage, 4),
        "view_coverage": round(measured.coverage, 4),
        "alignment_search": "none",
    }

    findings: list[dict[str, Any]] = []

    if (
        luminance_ratio < SUBJECT_LUMINANCE_RATIO_MIN
        and measured.subject_luminance_p90 < SUBJECT_LUMINANCE_P90_FLOOR
    ):
        findings.append(finding(
            VALUE_COLLAPSE,
            "error",
            f"The {view or 'synthesized'} view's subject is rendered at only "
            f"{luminance_ratio:.0%} of the front reference's brightness (mean "
            f"subject luminance {measured.subject_luminance:.1f} vs "
            f"{front.subject_luminance:.1f}) and carries no bright content at "
            f"all (p90 luminance {measured.subject_luminance_p90:.1f}) -- a "
            f"backlit or silhouetted figure, not a lit redraw of the character. "
            f"Shape checks cannot see this: a solid dark silhouette matches its "
            f"front's outline perfectly.",
            metric="subject_luminance_ratio",
            actual=round(luminance_ratio, 4),
            expected=(
                f">= {SUBJECT_LUMINANCE_RATIO_MIN} (or view p90 luminance "
                f">= {SUBJECT_LUMINANCE_P90_FLOOR:g})"
            ),
        ))

    front_shows_legs = front.gap_row_fraction >= FRONT_LEG_SEPARATION_MIN
    view_shows_legs = measured.gap_row_fraction >= VIEW_LEG_SEPARATION_MIN

    if front_shows_legs and not view_shows_legs:
        if plane == "front-plane":
            findings.append(finding(
                LEG_SEPARATION_MISMATCH,
                "error",
                f"The front reference shows separated legs across "
                f"{front.gap_row_fraction:.0%} of its lower body, but the "
                f"{view or 'synthesized'} view -- which shares the front's camera "
                f"plane, so nothing can occlude the separation -- shows it across "
                f"only {measured.gap_row_fraction:.0%}. The view has reinterpreted "
                f"the character's lower garment.",
                metric="view_gap_row_fraction",
                actual=round(measured.gap_row_fraction, 4),
                expected=f">= {VIEW_LEG_SEPARATION_MIN}",
            ))
        elif width_ratio > LOWER_WIDTH_RATIO_ERROR:
            findings.append(finding(
                LEG_SEPARATION_MISMATCH,
                "error",
                f"The front reference shows separated legs across "
                f"{front.gap_row_fraction:.0%} of its lower body; the "
                f"{view or 'synthesized'} view shows none and its lower body is "
                f"{width_ratio:.2f}x the front's width instead of narrowing the way "
                f"a pair of legs seen edge-on must. That is a floor-length garment, "
                f"not a profile of the reference's legs.",
                metric="lower_width_ratio",
                actual=round(width_ratio, 4),
                expected=f"<= {LOWER_WIDTH_RATIO_ERROR} when the leg gap is lost",
            ))

    if plane == "front-plane" and correlation < FRONT_PLANE_CORR_WARNING:
        findings.append(finding(
            LOWER_SILHOUETTE_DECORRELATED,
            "warning",
            f"The {view or 'synthesized'} view's lower-body width profile "
            f"correlates only {correlation:.2f} with the front reference's over "
            f"{PROFILE_BINS} bins, though it shares the front's camera plane. The "
            f"lower silhouette has changed shape.",
            metric="lower_profile_corr",
            actual=round(correlation, 4),
            expected=f">= {FRONT_PLANE_CORR_WARNING}",
        ))
    elif plane != "front-plane" and LOWER_WIDTH_RATIO_WARNING < width_ratio <= LOWER_WIDTH_RATIO_ERROR:
        # A profile view's front-vs-view width-profile CORRELATION is not
        # discriminating -- known-good profile views measure anywhere from
        # -0.09 to 0.98 against their own front, because a 90-degree turn
        # legitimately reshapes every row. So the warning tier for a profile
        # view is the sub-threshold widening instead, which is.
        findings.append(finding(
            LOWER_SILHOUETTE_DECORRELATED,
            "warning",
            f"The {view or 'synthesized'} view's lower body is {width_ratio:.2f}x "
            f"the front reference's width. A profile of the front's legs should be "
            f"narrower, not wider -- the garment may have been widened.",
            metric="lower_width_ratio",
            actual=round(width_ratio, 4),
            expected=f"<= {LOWER_WIDTH_RATIO_WARNING}",
        ))

    if any(item["severity"] == "error" for item in findings):
        status = "blocked"
    elif findings:
        status = "warning"
    else:
        status = "clean"

    return {
        "check_version": CHECK_VERSION,
        "view": view,
        "view_plane": plane,
        "evaluable": True,
        "status": status,
        "measurements": measurements,
        "findings": findings,
    }


def blocking_view_conflicts(results: dict[str, dict[str, Any]]) -> list[str]:
    """The error-grade conflicts a pre-generate gate must refuse to run on.

    `results` maps a view label to a `compare_view_to_front` result. One line
    per offending view, leading with the view's own name so an operator
    reading a 4xx knows which tile to fix without opening anything. Warning
    and not-evaluable findings never appear here -- they are advisory and ride
    on the job's warnings instead, because a check that blocks on its own
    uncertainty is a check operators learn to route around.
    """
    lines: list[str] = []
    for label in sorted(results):
        errors = [item for item in results[label].get("findings", []) if item["severity"] == "error"]
        if not errors:
            continue
        detail = " ".join(item["message"] for item in errors)
        lines.append(
            f"{label.upper()} view contradicts the front image — regenerate or "
            f"remove it. {detail}"
        )
    return lines


def job_warnings(results: dict[str, dict[str, Any]]) -> list[str]:
    """Roll per-view results up into operator-facing job warning lines.

    `results` maps a view label to a `compare_view_to_front` result. One line
    per finding, prefixed with the view it belongs to, so the studio's job
    warning list names the offending tile before an operator spends a
    generate on it."""
    lines: list[str] = []
    for label in sorted(results):
        for item in results[label].get("findings", []):
            lines.append(f"{label} view: {item['message']}")
    return lines
