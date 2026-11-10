from __future__ import annotations

from collections.abc import Iterable

from .proxy_rig import resolve_right_hand_bone


MAX_DEFORMATION_REVIEW_FRAMES = 600


def deformation_review_frame_specs() -> tuple[tuple[str, float, bool, str | None], ...]:
    """Return the deterministic full-body and focused animation review set."""
    return (
        ("rest", 0.0, True, None),
        ("attack-00", 0.0, False, None),
        ("attack-12", 0.125, False, None),
        ("attack-25", 0.25, False, None),
        ("attack-38", 0.375, False, None),
        ("attack-50", 0.5, False, None),
        ("weapon-hand-50", 0.5, False, "right-hand"),
        ("attack-62", 0.625, False, None),
        ("attack-75", 0.75, False, None),
        ("attack-88", 0.875, False, None),
        ("attack-100", 1.0, False, None),
    )


def resolve_review_focus_bone(available: Iterable[str], focus: str) -> str:
    """Resolve semantic review targets while preserving explicit bone names."""
    names = list(available)
    if focus == "right-hand":
        return resolve_right_hand_bone(names)
    if focus not in names:
        raise ValueError(f"Review focus bone {focus!r} is not present")
    return focus


def select_review_frame(frame_start: int, frame_end: int, ratio: float) -> int:
    if frame_end < frame_start:
        raise ValueError("frame_end must be greater than or equal to frame_start")
    bounded_ratio = min(1.0, max(0.0, float(ratio)))
    return round(frame_start + (frame_end - frame_start) * bounded_ratio)


def all_animation_frames(frame_start: int, frame_end: int) -> range:
    if frame_end < frame_start:
        raise ValueError("frame_end must be greater than or equal to frame_start")
    frame_count = frame_end - frame_start + 1
    if frame_count > MAX_DEFORMATION_REVIEW_FRAMES:
        raise ValueError(
            f"Animation contains {frame_count} frames; deformation review is limited to "
            f"{MAX_DEFORMATION_REVIEW_FRAMES}"
        )
    return range(frame_start, frame_end + 1)
