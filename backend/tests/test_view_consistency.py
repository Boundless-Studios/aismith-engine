"""Tests for the view-consistency check in `backend.app.view_consistency`.

Two layers:

* Synthetic silhouettes exercise every branch deterministically -- legs vs
  skirt, front-plane vs profile, and every refuse-to-score guard.
* Rendered fixtures (256px Blender renders of CC0 character models, made by
  `scripts/render_cc0_fixtures.py`) reproduce the failure that motivated the
  module and a known-good view set, so a future threshold change cannot
  silently stop catching it.
"""
from __future__ import annotations

import asyncio
import json
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from backend.app.view_consistency import (
    LEG_SEPARATION_MISMATCH,
    LOWER_SILHOUETTE_DECORRELATED,
    NOT_EVALUABLE,
    VALUE_COLLAPSE,
    blocking_view_conflicts,
    compare_view_to_front,
    job_warnings,
    measure_silhouette,
    view_plane,
)

FIXTURES = Path(__file__).parent / "fixtures" / "view_consistency"

BACKGROUND = (200, 200, 200)
SUBJECT = (40, 40, 60)


def _png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _figure(
    *,
    size: int = 400,
    head_top: int = 20,
    hip: int = 260,
    foot: int = 380,
    torso_half_width: int = 40,
    leg_half_width: int = 14,
    leg_offset: int = 26,
    skirt_half_width: int | None = None,
    background=BACKGROUND,
    subject=SUBJECT,
) -> bytes:
    """A crude standing figure on a plain background.

    With `skirt_half_width` the lower body is one solid trapezoid (a robe);
    without it, two separated legs straddling the centre line. `leg_offset` of
    0 collapses the two legs into a single column, which is what a 90-degree
    profile view of a two-legged character looks like.
    """
    image = Image.new("RGB", (size, size), background)
    pixels = image.load()
    centre = size // 2
    for y in range(head_top, hip):
        for x in range(centre - torso_half_width, centre + torso_half_width):
            pixels[x, y] = subject
    for y in range(hip, foot):
        if skirt_half_width is not None:
            flare = skirt_half_width * (0.6 + 0.4 * (y - hip) / max(1, foot - hip))
            spans = [(centre - int(flare), centre + int(flare))]
        elif leg_offset == 0:
            spans = [(centre - leg_half_width, centre + leg_half_width)]
        else:
            spans = [
                (centre - leg_offset - leg_half_width, centre - leg_offset + leg_half_width),
                (centre + leg_offset - leg_half_width, centre + leg_offset + leg_half_width),
            ]
        for start, end in spans:
            for x in range(start, end):
                pixels[x, y] = subject
    return _png(image)


class ValueCollapseTests(unittest.TestCase):
    """A backlit/silhouetted view must trip the luminance check -- shape
    signals are structurally blind to it."""

    LIT = (150, 148, 152)
    DARK = (12, 12, 14)

    def test_a_near_black_profile_of_a_lit_front_is_blocked(self):
        front = _figure(subject=self.LIT)
        silhouette_view = _figure(leg_offset=0, subject=self.DARK)

        result = compare_view_to_front(front, silhouette_view, view="left")

        codes = [item["code"] for item in result["findings"]]
        self.assertIn(VALUE_COLLAPSE, codes)
        self.assertEqual(result["status"], "blocked")
        self.assertLess(result["measurements"]["subject_luminance_ratio"], 0.2)

    def test_an_equally_lit_profile_carries_no_value_finding(self):
        front = _figure(subject=self.LIT)
        lit_view = _figure(leg_offset=0, subject=self.LIT)

        result = compare_view_to_front(front, lit_view, view="left")

        codes = [item["code"] for item in result["findings"]]
        self.assertNotIn(VALUE_COLLAPSE, codes)
        self.assertGreater(result["measurements"]["subject_luminance_ratio"], 0.9)

    def test_a_dark_clothed_character_with_real_highlights_is_not_penalised(self):
        # A dark-coated character's known-good left profile in the
        # calibration set measured a mean ratio of 0.53 (the coat dominates
        # the view) yet p90 luminance 97 -- genuine highlights. Mean ratio
        # alone cannot separate that from a silhouette; the p90 floor must
        # let it through.
        front = _figure(subject=self.LIT)
        dark_view = Image.open(BytesIO(_figure(leg_offset=0, subject=(30, 30, 32))))
        pixels = dark_view.load()
        # Paint a lit face/chest band: enough bright subject pixels to lift
        # p90 above the floor while the mean stays well under the ratio line.
        # NOT near the 200-gray backdrop: a highlight within the background
        # key's tolerance would be keyed as background and excluded from the
        # subject mask entirely (which is how the first draft of this test
        # accidentally deleted its own highlights).
        for y in range(20, 80):
            for x in range(160, 240):
                if pixels[x, y] == (30, 30, 32):
                    pixels[x, y] = (250, 226, 130)
        result = compare_view_to_front(front, _png(dark_view), view="left")

        codes = [item["code"] for item in result["findings"]]
        self.assertNotIn(VALUE_COLLAPSE, codes)
        self.assertLess(result["measurements"]["subject_luminance_ratio"], 0.55)

    def test_a_uniformly_dark_character_is_not_penalised(self):
        # Ratio, not absolute: a legitimately dark-styled character whose
        # front is equally dark must pass.
        front = _figure(subject=self.DARK)
        view = _figure(leg_offset=0, subject=self.DARK)

        result = compare_view_to_front(front, view, view="left")

        codes = [item["code"] for item in result["findings"]]
        self.assertNotIn(VALUE_COLLAPSE, codes)


class SyntheticSilhouetteTests(unittest.TestCase):
    """Every branch, on shapes whose geometry is known by construction."""

    def test_a_two_legged_front_measures_a_gap_in_most_lower_rows(self):
        measured = measure_silhouette(_figure())

        self.assertIsNotNone(measured)
        # The band is the bottom 40% of the subject; this figure's hips sit a
        # little above that line, so the band's top ~17% is solid torso and
        # the rest is legs. Everything below the hips carries a gap.
        self.assertGreater(measured.gap_row_fraction, 0.8)
        self.assertLess(measured.gap_row_fraction, 1.0)

    def test_a_skirt_measures_no_gap_at_all(self):
        measured = measure_silhouette(_figure(skirt_half_width=60))

        self.assertIsNotNone(measured)
        self.assertEqual(measured.gap_row_fraction, 0.0)

    def test_a_matching_back_view_of_a_two_legged_character_is_clean(self):
        front = _figure()
        back = _figure(leg_offset=27, leg_half_width=15)

        result = compare_view_to_front(front, back, view="back")

        self.assertEqual(result["status"], "clean")
        self.assertEqual(result["findings"], [])

    def test_a_back_view_that_fuses_the_legs_into_a_skirt_is_an_error(self):
        result = compare_view_to_front(_figure(), _figure(skirt_half_width=60), view="back")

        self.assertEqual(result["status"], "blocked")
        codes = [item["code"] for item in result["findings"]]
        self.assertIn(LEG_SEPARATION_MISMATCH, codes)
        error = next(item for item in result["findings"] if item["code"] == LEG_SEPARATION_MISMATCH)
        self.assertEqual(error["severity"], "error")
        self.assertEqual(error["metric"], "view_gap_row_fraction")

    def test_a_profile_view_of_the_same_legs_loses_the_gap_and_stays_clean(self):
        """The whole reason gap-loss alone cannot be the rule: a 90-degree
        profile of two separated legs shows one narrow column, no gap."""
        front = _figure()
        profile = _figure(leg_offset=0, leg_half_width=16, torso_half_width=30)

        measured = measure_silhouette(profile)
        self.assertEqual(measured.gap_row_fraction, 0.0)

        result = compare_view_to_front(front, profile, view="left")

        self.assertEqual(result["status"], "clean")
        self.assertLess(result["measurements"]["lower_width_ratio"], 1.0)

    def test_a_profile_view_that_is_a_robe_is_an_error(self):
        result = compare_view_to_front(_figure(), _figure(skirt_half_width=60), view="left")

        self.assertEqual(result["status"], "blocked")
        error = next(item for item in result["findings"] if item["code"] == LEG_SEPARATION_MISMATCH)
        self.assertEqual(error["metric"], "lower_width_ratio")
        self.assertGreater(error["actual"], 1.25)

    def test_a_consistently_skirted_character_never_trips_the_check(self):
        """Design invariant, verified rather than asserted: the check compares
        AGAINST the front, so a character who wears a floor-length skirt in
        the front too has no leg separation for any view to contradict."""
        skirted_front = _figure(skirt_half_width=60)

        for label in ("back", "left", "right"):
            with self.subTest(view=label):
                result = compare_view_to_front(
                    skirted_front, _figure(skirt_half_width=58), view=label
                )
                codes = [item["code"] for item in result["findings"]]
                self.assertNotIn(LEG_SEPARATION_MISMATCH, codes)

    def test_an_unlabelled_view_takes_the_conservative_profile_branch(self):
        """An unknown label must never be condemned on gap loss alone."""
        self.assertEqual(view_plane("sideways"), "unknown")

        result = compare_view_to_front(
            _figure(), _figure(leg_offset=0, leg_half_width=16), view="sideways"
        )

        self.assertEqual(result["status"], "clean")

    def test_a_front_plane_view_whose_lower_profile_decorrelates_warns(self):
        front = _figure()
        # Legs still separated -- so no leg_separation_mismatch -- but much
        # thicker and set closer together, which inverts the per-row width
        # profile relative to the front's.
        back = _figure(leg_half_width=24, leg_offset=30)

        result = compare_view_to_front(front, back, view="back")

        self.assertEqual(result["status"], "warning")
        warning = next(
            item for item in result["findings"] if item["code"] == LOWER_SILHOUETTE_DECORRELATED
        )
        self.assertEqual(warning["severity"], "warning")
        # A front-plane comparison warns on correlation, never on width ratio:
        # correlation is the signal that discriminates in this plane.
        self.assertEqual(warning["metric"], "lower_profile_corr")
        self.assertLess(warning["actual"], 0.85)

    def test_a_profile_view_widened_below_the_error_gate_warns(self):
        front = _figure()
        widened = _figure(leg_offset=0, leg_half_width=32)

        result = compare_view_to_front(front, widened, view="left")

        self.assertEqual(result["status"], "warning")
        warning = next(
            item for item in result["findings"] if item["code"] == LOWER_SILHOUETTE_DECORRELATED
        )
        self.assertEqual(warning["metric"], "lower_width_ratio")
        self.assertGreater(warning["actual"], 1.05)
        self.assertLessEqual(warning["actual"], 1.25)


class RefusalTests(unittest.TestCase):
    """Refuse to score rather than degrade to a weaker comparison."""

    def test_a_non_uniform_background_is_not_evaluable(self):
        gradient = Image.new("RGB", (400, 400))
        pixels = gradient.load()
        for y in range(400):
            for x in range(400):
                pixels[x, y] = (120 + y // 4, 120 + y // 4, 120 + y // 4)

        result = compare_view_to_front(_figure(), _png(gradient), view="left")

        self.assertEqual(result["status"], "not_evaluable")
        self.assertFalse(result["evaluable"])
        self.assertEqual(result["findings"][0]["code"], NOT_EVALUABLE)
        self.assertEqual(result["findings"][0]["severity"], "not-evaluable")
        self.assertEqual(result["measurements"], {})

    def test_a_frame_filling_subject_is_not_evaluable(self):
        # A plain frame with a near-invisible border: uniform enough to key,
        # but the key then claims essentially the whole frame as subject.
        solid = Image.new("RGB", (400, 400), (40, 40, 60))
        for x in range(400):
            for y in list(range(4)) + list(range(396, 400)):
                solid.putpixel((x, y), BACKGROUND)
                solid.putpixel((y, x), BACKGROUND)

        result = compare_view_to_front(_figure(), _png(solid), view="left")

        self.assertEqual(result["status"], "not_evaluable")

    def test_an_empty_frame_is_not_evaluable(self):
        blank = Image.new("RGB", (400, 400), BACKGROUND)

        result = compare_view_to_front(_figure(), _png(blank), view="left")

        self.assertEqual(result["status"], "not_evaluable")

    def test_a_constant_width_lower_body_is_not_evaluable(self):
        """Correlation is undefined against a profile with no variance, so
        the pair is refused rather than scored on an arbitrary value."""
        result = compare_view_to_front(_figure(), _figure(hip=200), view="back")

        self.assertEqual(result["status"], "not_evaluable")
        self.assertEqual(result["findings"][0]["code"], NOT_EVALUABLE)

    def test_a_bad_front_reference_refuses_before_measuring_the_view(self):
        blank = Image.new("RGB", (400, 400), BACKGROUND)

        result = compare_view_to_front(_png(blank), _figure(), view="left")

        self.assertEqual(result["status"], "not_evaluable")
        self.assertIn("front reference", result["findings"][0]["message"])


class ReportingTests(unittest.TestCase):
    def test_every_score_is_published_with_its_denominator(self):
        result = compare_view_to_front(_figure(), _figure(), view="back")
        measurements = result["measurements"]

        for key in (
            "front_band_rows",
            "view_band_rows",
            "profile_bins",
            "front_coverage",
            "view_coverage",
        ):
            self.assertIn(key, measurements)
        self.assertGreater(measurements["front_band_rows"], 0)
        self.assertEqual(measurements["profile_bins"], 64)
        # No alignment/offset search exists, so no optimum can lock to a bound.
        self.assertEqual(measurements["alignment_search"], "none")

    def test_job_warnings_name_the_offending_view(self):
        results = {
            "left": compare_view_to_front(_figure(), _figure(skirt_half_width=60), view="left"),
            "right": compare_view_to_front(_figure(), _figure(), view="right"),
        }

        lines = job_warnings(results)

        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("left view: "))

    def test_a_clean_pair_produces_no_job_warnings(self):
        results = {"back": compare_view_to_front(_figure(), _figure(), view="back")}

        self.assertEqual(job_warnings(results), [])

    def test_blocking_conflicts_name_the_view_and_the_remedy(self):
        results = {
            "left": compare_view_to_front(_figure(), _figure(skirt_half_width=60), view="left"),
            "right": compare_view_to_front(_figure(), _figure(), view="right"),
        }

        lines = blocking_view_conflicts(results)

        self.assertEqual(len(lines), 1)
        self.assertTrue(
            lines[0].startswith("LEFT view contradicts the front image"), lines[0]
        )
        self.assertIn("regenerate or remove it", lines[0])

    def test_a_warning_grade_finding_never_blocks(self):
        # The pre-generate gate blocks on error-grade findings only. A warning
        # is advisory by construction; promoting it to a block would teach
        # operators to route around the check.
        warned = {
            "check_version": 2,
            "view": "back",
            "view_plane": "front-plane",
            "evaluable": True,
            "status": "warning",
            "measurements": {},
            "findings": [{
                "code": LOWER_SILHOUETTE_DECORRELATED,
                "severity": "warning",
                "message": "The back view's lower-body width profile correlates only 0.80.",
                "metric": "lower_profile_corr",
                "actual": 0.8,
                "expected": ">= 0.85",
            }],
        }

        self.assertEqual(blocking_view_conflicts({"back": warned}), [])
        self.assertEqual(len(job_warnings({"back": warned})), 1)

    def test_a_not_evaluable_result_never_blocks(self):
        refused = compare_view_to_front(
            _figure(background=(200, 200, 200)),
            _png(Image.new("RGB", (400, 400), (200, 200, 200))),
            view="left",
        )

        self.assertEqual(refused["status"], "not_evaluable")
        self.assertEqual(blocking_view_conflicts({"left": refused}), [])


class RenderedFixtureTests(unittest.TestCase):
    """CC0 renders that reproduce the motivating failure, 256px.

    Blender renders of CC0 models vendored with Mesh2Motion; sources,
    licences and hashes are in `fixtures/view_consistency/PROVENANCE.md`.
    `front-legs` is Quaternius's Female Mannequin in a T-pose, dressed in a
    fitted jumpsuit and boots, with two clearly separated legs.
    `left-robe-drift` and `back-robe-drift` are the same figure in a
    floor-length robe -- the reinterpretation that view synthesis produced
    and that the mesh stage then fused into a skirt.
    `front-zombie`/`left-zombie-good`/`back-zombie-good` are a consistent
    known-good set of a different two-legged character (Kenney's Zombie).
    `front-robe-source` is the robed figure's own front -- the source a
    robed view would legitimately be drawn from -- which is what makes the
    skirted-character invariant testable on an image rather than a shape.
    """

    def _fixture(self, name: str) -> bytes:
        path = FIXTURES / f"{name}.png"
        self.assertTrue(path.is_file(), f"missing fixture {path}")
        return path.read_bytes()

    def test_the_left_view_robe_drift_is_an_error(self):
        result = compare_view_to_front(
            self._fixture("front-legs"), self._fixture("left-robe-drift"), view="left"
        )

        self.assertEqual(result["status"], "blocked")
        codes = [item["code"] for item in result["findings"]]
        self.assertIn(LEG_SEPARATION_MISMATCH, codes)
        self.assertGreater(result["measurements"]["lower_width_ratio"], 1.25)

    def test_the_back_view_robe_drift_is_an_error(self):
        result = compare_view_to_front(
            self._fixture("front-legs"), self._fixture("back-robe-drift"), view="back"
        )

        self.assertEqual(result["status"], "blocked")
        codes = [item["code"] for item in result["findings"]]
        self.assertIn(LEG_SEPARATION_MISMATCH, codes)

    def test_the_known_good_zombie_views_stay_clean(self):
        for view, name in (("left", "left-zombie-good"), ("back", "back-zombie-good")):
            with self.subTest(view=view):
                result = compare_view_to_front(
                    self._fixture("front-zombie"), self._fixture(name), view=view
                )
                self.assertEqual(result["status"], "clean", result["findings"])

    def test_the_same_drifted_view_is_clean_against_the_robed_front_it_was_drawn_from(self):
        """The skirted-character design point, on rendered images: the check
        compares AGAINST the front. The identical left view is an error
        against the jumpsuit front and clean against the robed front --
        because that front shows no leg separation to contradict."""
        result = compare_view_to_front(
            self._fixture("front-robe-source"), self._fixture("left-robe-drift"), view="left"
        )

        self.assertEqual(result["status"], "clean", result["findings"])
        self.assertLess(result["measurements"]["front"]["gap_row_fraction"], 0.35)


class SynthesizeViewsWiringTests(unittest.TestCase):
    """The synthesize-views job records findings on every view artifact and
    rolls them up into the job's warnings."""

    def _run(self, view_bytes_by_label):
        from backend.app import main

        submitted: list[str] = []

        def fake_edit(source_path, output_path, **kwargs):
            label = Path(output_path).stem
            submitted.append(label)
            Path(output_path).write_bytes(view_bytes_by_label[label])
            return {"prompt_id": f"prompt-{label}", "model": "test"}

        with TemporaryDirectory() as directory:
            artifacts = Path(directory)
            source_png = artifacts / "src-1.png"
            source_png.write_bytes(_figure())
            (artifacts / "src-1.json").write_text(json.dumps({
                "id": "src-1",
                "name": "Prepared image.png",
                "kind": "image",
                "path": str(source_png),
                "size_bytes": source_png.stat().st_size,
                "created_at": 0.0,
                "metadata": {"stage": "prepare-image"},
            }))
            job = main.JobState(
                id="job-vc-1",
                kind="synthesize-views",
                submission_id="11111111-1111-1111-1111-111111111111",
                source_artifact_id="src-1",
            )
            request = main.JobRequest(
                kind="synthesize-views",
                source_artifact_id="src-1",
                submission_id="11111111-1111-1111-1111-111111111111",
                settings={"views": list(view_bytes_by_label)},
            )
            with (
                patch.object(main, "ARTIFACTS", artifacts),
                patch.object(main, "run_comfyui_image_edit", fake_edit),
            ):
                asyncio.run(main._execute_job(job, request))
                saved = {}
                for path in artifacts.glob("*.json"):
                    payload = json.loads(path.read_text())
                    saved[payload["id"]] = payload
            return job, saved, submitted

    def test_a_drifted_view_is_recorded_on_the_artifact_and_warned_on_the_job(self):
        job, saved, submitted = self._run({
            "left": _figure(skirt_half_width=60),
            "back": _figure(),
        })

        self.assertEqual(job.status, "completed", job.error)
        self.assertEqual(submitted, ["left", "back"])

        primary = saved[job.output_artifact_id]["metadata"]
        self.assertEqual(primary["view_consistency"]["status"], "blocked")
        self.assertEqual(
            [item["code"] for item in primary["view_consistency"]["findings"]],
            [LEG_SEPARATION_MISMATCH],
        )
        self.assertEqual(primary["view_consistency_summary"]["left"]["status"], "blocked")
        self.assertEqual(primary["view_consistency_summary"]["back"]["status"], "clean")

        sibling = next(
            payload for artifact_id, payload in saved.items()
            if artifact_id not in {"src-1", job.output_artifact_id}
        )
        self.assertEqual(sibling["metadata"]["view_consistency"]["status"], "clean")

        self.assertTrue(
            any(line.startswith("left view: ") for line in job.warnings),
            job.warnings,
        )

    def test_a_consistent_view_set_adds_no_consistency_warnings(self):
        job, saved, _ = self._run({"back": _figure(), "left": _figure(leg_offset=0, leg_half_width=16)})

        self.assertEqual(job.status, "completed", job.error)
        primary = saved[job.output_artifact_id]["metadata"]
        self.assertEqual(primary["view_consistency"]["status"], "clean")
        self.assertEqual(
            {label: entry["status"] for label, entry in primary["view_consistency_summary"].items()},
            {"back": "clean", "left": "clean"},
        )
        self.assertFalse([line for line in job.warnings if " view: " in line], job.warnings)


if __name__ == "__main__":
    unittest.main()
