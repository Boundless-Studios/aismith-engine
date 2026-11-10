"""S2 project-views conformance fixture (task5): a Blender-executed test that
builds a synthetic textured mesh, renders its four conditioning views with
the SAME camera conventions `project_views` uses, runs project-views end to
end, and asserts numerically on depth agreement, camera-facing pass-rate
(including the thin arm plates), colour agreement and background rejection.
Replaces render-and-look verification of the S2 stage.
"""
import json
import os
import re
import subprocess
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from backend.app import native, view_projection

ROLES = ("front", "back", "left", "right")


def require_blender_4x() -> None:
    """Skip unless a Blender 4.x is available: the compositor File Output
    node never fires during a background render on Blender 5.x (upstream
    #150625), so project-views' depth pass cannot be produced there. The GPU
    box that runs these fixtures for real is on 4.2."""
    blender = native.find_blender()
    if blender is None:
        raise unittest.SkipTest("Blender not found (BLENDER_EXE / PATH)")
    version = subprocess.run(
        [str(blender), "--version"], capture_output=True, text=True, timeout=120, check=False,
    ).stdout
    major = re.search(r"Blender (\d+)\.", version)
    if major is None or int(major.group(1)) >= 5:
        raise unittest.SkipTest(f"project-views depth pass needs Blender 4.x (found: {version.strip()[:40]!r})")


class ProjectionConformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_blender_4x()
        cls.tmp_dir = Path(tempfile.mkdtemp(prefix="aismith-projection-conformance-"))
        fixture_glb = cls.tmp_dir / "fixture.glb"
        cls.atlas_png = cls.tmp_dir / "atlas.png"
        meta_json = cls.tmp_dir / "fixture_meta.json"
        cls.fixture_meta = native.build_projection_fixture(
            fixture_glb, cls.atlas_png, meta_json, atlas_size=512,
        )

        views_dir = cls.tmp_dir / "views"
        views = native.render_role_views(fixture_glb, views_dir, roles=ROLES, size=512)

        cls.out_glb = cls.tmp_dir / "out.glb"
        with patch.dict(os.environ, {"AISMITH_PROJECT_VIEWS_DEBUG": "1"}):
            cls.report = native.run_project_views(
                fixture_glb, cls.out_glb, views, {"cycles_samples": 8},
            )
        cls.render_dir = cls.out_glb.parent / "project-views-render"

        cls.role_data = {}
        for role in ROLES:
            with np.load(cls.render_dir / f"{role}_texels.npz") as npz:
                texels = {key: npz[key] for key in npz.files}
            cls.role_data[role] = {
                "texels": texels,
                "depth_raw": np.load(cls.render_dir / f"{role}_depth.npy"),
                "alpha": np.load(cls.render_dir / f"{role}_alpha.npy"),
                "meta": json.loads((cls.render_dir / f"{role}_meta.json").read_text(encoding="utf-8")),
            }
        cls.projected_mask = np.load(cls.render_dir / "projected_mask.npy")
        cls.atlas_out = np.load(cls.render_dir / "atlas_out.npy")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def _facing_bound_alpha_hit(self, role):
        """T: covered texels facing the camera, inside its render, whose
        render pixel actually hit geometry -- the population every depth
        assertion below is scoped to."""
        data = self.role_data[role]["texels"]
        depth_raw = self.role_data[role]["depth_raw"]
        alpha = self.role_data[role]["alpha"]
        meta = self.role_data[role]["meta"]
        # Indices are only meaningful where in_render_bounds is True; clip
        # for safe fancy-indexing everywhere else, then mask them out via T.
        ix = np.clip(data["render_ix"], 0, meta["image_width"] - 1)
        iy = np.clip(data["render_iy"], 0, meta["image_height"] - 1)
        alpha_at = alpha[iy, ix]
        T = data["in_render_bounds"] & (data["facing"] > 0.5) & (alpha_at > 0.5)
        return data, depth_raw, ix, iy, T

    def _at_depth_discontinuity(self, role, ix, iy):
        """Per-texel: does the render pixel's 3x3 neighbourhood span more than
        `depth_bias` (an occlusion contour or silhouette)? The one-sided
        neighbourhood-minimum test rejects a one-pixel band there BY DESIGN
        (it may never manufacture a false visible), so depth assertions
        judge the remaining texels -- where a sentinel or a systematic
        offset would show -- and only bound the band's size."""
        depth_raw = self.role_data[role]["depth_raw"]
        alpha = self.role_data[role]["alpha"]
        depth_bias = self.role_data[role]["meta"]["depth_bias"]
        valid = alpha > 0.5
        padded_lo = np.pad(np.where(valid, depth_raw, np.inf), 1, mode="edge")
        padded_hi = np.pad(np.where(valid, depth_raw, -np.inf), 1, mode="edge")
        h, w = depth_raw.shape
        windows_lo = np.stack([padded_lo[i:i + h, j:j + w] for i in range(3) for j in range(3)])
        windows_hi = np.stack([padded_hi[i:i + h, j:j + w] for i in range(3) for j in range(3)])
        span = windows_hi.max(axis=0) - windows_lo.min(axis=0)
        return span[iy, ix] > depth_bias

    def test_depth_map_agrees_with_analytic_texel_depth(self):
        for role in ROLES:
            with self.subTest(role=role):
                data, depth_raw, ix, iy, T = self._facing_bound_alpha_hit(role)
                meta = self.role_data[role]["meta"]
                camera_distance, diagonal, depth_bias = (
                    meta["camera_distance"], meta["diagonal"], meta["depth_bias"],
                )
                self.assertGreater(int(T.sum()), 0, f"{role}: empty T population")

                depth = data["depth"]
                delta = (depth_raw[iy, ix] - camera_distance) - depth
                median_abs = float(np.median(np.abs(delta[T])))
                frac_within = float(np.mean(np.abs(delta[T]) <= depth_bias))
                smooth = T & ~self._at_depth_discontinuity(role, ix, iy)
                frac_within_smooth = float(np.mean(np.abs(delta[smooth]) <= depth_bias))

                # Diagnostics only, computed on failure's behalf so a
                # convention bug is visible in the message: a row-order bug
                # (flipud) and a sign-convention bug (-delta) each leave a
                # distinct fingerprint here.
                flipped = np.flipud(depth_raw)
                delta_flipped = (flipped[iy, ix] - camera_distance) - depth
                median_abs_flipped = float(np.median(np.abs(delta_flipped[T])))
                frac_within_flipped = float(np.mean(np.abs(delta_flipped[T]) <= depth_bias))
                delta_negated = -delta
                median_abs_negated = float(np.median(np.abs(delta_negated[T])))
                frac_within_negated = float(np.mean(np.abs(delta_negated[T]) <= depth_bias))

                message = (
                    f"{role}: median|delta|={median_abs:.6f} (tol {1e-3 * diagonal:.6f}), "
                    f"frac_within_bias={frac_within:.4f} (n={int(T.sum())}, bias={depth_bias:.6f}), "
                    f"away from discontinuities={frac_within_smooth:.4f} (n={int(smooth.sum())}) | "
                    f"diag flipud(depth_raw): median|delta|={median_abs_flipped:.6f}, "
                    f"frac_within={frac_within_flipped:.4f} | "
                    f"diag -delta: median|delta|={median_abs_negated:.6f}, "
                    f"frac_within={frac_within_negated:.4f}"
                )
                self.assertLessEqual(median_abs, 1e-3 * diagonal, message)
                # The mirrored-camera bug (2026-09-05) read 0.67 / 0.21 (head)
                # here; the contour band alone costs ~1% at 512 px.
                self.assertGreaterEqual(frac_within, 0.98, message)
                self.assertGreaterEqual(frac_within_smooth, 0.99, message)

    def test_background_pixels_never_seed_the_depth_minimum(self):
        for role in ROLES:
            with self.subTest(role=role):
                data, depth_raw, ix, iy, T = self._facing_bound_alpha_hit(role)
                alpha = self.role_data[role]["alpha"]
                depth_bias = self.role_data[role]["meta"]["depth_bias"]
                self.assertGreater(int(T.sum()), 0, f"{role}: empty T population")

                depth = data["depth"]
                depth_at_render = data["depth_at_render"]
                pulled_nearer = depth_at_render < depth - depth_bias
                frac_ok = float(np.mean(~pulled_nearer[T]))
                at_contour = self._at_depth_discontinuity(role, ix, iy)
                # A texel the 3x3 minimum pulls nearer AWAY from any contour
                # can only be a background sentinel winning the minimum or a
                # systematic offset -- the two failure modes this guards.
                frac_bad_smooth = float(np.mean(pulled_nearer[T & ~at_contour]))

                background = depth_raw[alpha <= 0.5]
                sentinel_min = float(background.min()) if background.size else float("nan")
                sentinel_max = float(background.max()) if background.size else float("nan")
                message = (
                    f"{role}: frac(depth_at_render >= depth - bias)={frac_ok:.4f} (n={int(T.sum())}); "
                    f"pulled nearer away from contours={frac_bad_smooth:.4f} "
                    f"(n={int((T & ~at_contour).sum())}); "
                    f"background depth_raw sentinel range on this machine=[{sentinel_min:.4f}, {sentinel_max:.4f}]"
                )
                self.assertGreaterEqual(frac_ok, 0.95, message)
                self.assertLessEqual(frac_bad_smooth, 0.01, message)

    def test_camera_facing_texels_pass_the_depth_test(self):
        for role in ROLES:
            with self.subTest(role=role):
                data = self.role_data[role]["texels"]
                meta = self.role_data[role]["meta"]
                sample_ix = np.round(data["sample_px"]).astype(np.int64)
                sample_iy = np.round(data["sample_py"]).astype(np.int64)
                in_frame = (
                    (sample_ix >= 0) & (sample_ix < meta["image_width"])
                    & (sample_iy >= 0) & (sample_iy < meta["image_height"])
                )
                facing_positive = data["facing"] > 0.5
                F = data["in_render_bounds"] & in_frame & facing_positive
                self.assertGreater(int(F.sum()), 0, f"{role}: empty F population")

                pass_rate = float(np.mean(data["within_depth"][F]))
                message = f"{role}: overall pass-rate={pass_rate:.4f} (n={int(F.sum())})"
                self.assertGreaterEqual(pass_rate, 0.95, message)

                if role in ("front", "back"):
                    positions = data["positions"]
                    arm = (
                        (np.abs(positions[:, 0]) > 0.25)
                        & (positions[:, 2] > 0.95) & (positions[:, 2] < 1.05)
                    )
                    F_arm = F & arm
                    self.assertGreater(int(F_arm.sum()), 0, f"{role}: empty arm-plate F population")
                    pass_rate_arm = float(np.mean(data["within_depth"][F_arm]))
                    message_arm = f"{role} arm plates: pass-rate={pass_rate_arm:.4f} (n={int(F_arm.sum())})"
                    self.assertGreaterEqual(pass_rate_arm, 0.95, message_arm)

    def test_projected_colours_match_the_fixture_atlas(self):
        fixture_image = Image.open(self.atlas_png).convert("RGB")
        # atlas_png is written top-down (a conventional PNG file); atlas_out
        # is bottom-up (view_projection.py's convention, matched to Blender's
        # own image.pixels order) -- verified against the fixture's own known
        # axis-colour key (see build_projection_fixture) that this flip lines
        # the two up, not the un-flipped orientation.
        atlas_fixture = np.flipud(np.asarray(fixture_image, dtype=np.float64) / 255.0)
        mask = self.projected_mask
        self.assertGreater(int(mask.sum()), 0, "projected_mask selects no texels")

        # The fixture is self-emissive under a black world, so a rendered
        # pixel IS its atlas colour and the projection must reproduce the
        # atlas up to bilinear blur at colour-region boundaries. Judged on
        # texels at least one view sees well (max facing >= 0.5): a mirrored
        # or offset view sends whole faces and sphere halves to the wrong
        # colour there (0.897 with the mirror bug). Texels only ever seen at
        # grazing angles (the sphere's top cap under zero-tilt cameras) blend
        # foreshortened samples from several regions -- a known op limitation
        # (the 09-04 coat streaks), reported here, not asserted.
        palette = np.array(list(self.fixture_meta["face_colours"].values()), dtype=np.float64)
        texels = self.role_data["front"]["texels"]
        max_facing = np.max(np.stack([self.role_data[role]["texels"]["facing"] for role in ROLES]), axis=0)
        well_seen = np.zeros(mask.size, dtype=bool)
        well_seen[texels["covered_index"]] = max_facing >= 0.5
        well_seen = well_seen.reshape(mask.shape) & mask

        def classify(values):
            return np.argmin(np.linalg.norm(values[:, None, :] - palette[None, :, :], axis=-1), axis=1)

        # Bilinear sampling blurs one render pixel across a colour-region
        # boundary; the sphere packs ~7 atlas texels per view pixel (18% of
        # the atlas, 2.5% of the image), so that 1-px band is ~7 texels wide
        # in the atlas (measured: 9k texels, all on the sphere). Interior
        # texels -- further than BOUNDARY_RADIUS from any boundary of the
        # fixture's own label map -- must classify exactly.
        label_map = classify(atlas_fixture.reshape(-1, 3)).reshape(atlas_fixture.shape[:2])
        radius = 8
        padded = np.pad(label_map, radius, mode="edge")
        interior = np.ones(label_map.shape, dtype=bool)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                shifted = padded[radius + dy:radius + dy + label_map.shape[0], radius + dx:radius + dx + label_map.shape[1]]
                interior &= shifted == label_map
        judged = well_seen & interior

        accuracy_all = float(np.mean(classify(self.atlas_out[mask]) == classify(atlas_fixture[mask])))
        out, expected = self.atlas_out[judged], atlas_fixture[judged]
        accuracy = float(np.mean(classify(out) == classify(expected)))
        diff = np.abs(out - expected)
        median = float(np.median(diff))
        message = (
            f"well-seen interior texels: classification accuracy={accuracy:.4f}, median|diff|={median:.4f}, "
            f"p95|diff|={float(np.percentile(diff, 95)):.4f} (n={int(judged.sum())}); "
            f"all projected texels: accuracy={accuracy_all:.4f} (n={int(mask.sum())}, "
            f"boundary band={int((well_seen & ~interior).sum())})"
        )
        self.assertGreater(int(judged.sum()), 20000, message)
        self.assertGreaterEqual(accuracy, 0.99, message)
        self.assertLessEqual(median, 0.05, message)

    def test_no_texel_samples_the_background(self):
        all_weights = np.concatenate([self.role_data[role]["texels"]["weights"] for role in ROLES])
        all_rgb = np.concatenate([self.role_data[role]["texels"]["sample_rgb"] for role in ROLES], axis=0)
        contributing = all_weights > 0.0
        self.assertGreater(int(contributing.sum()), 0, "no texel carries positive weight in any role")

        luminance = all_rgb[contributing] @ np.array([0.2126, 0.7152, 0.0722])
        dark_fraction = float(np.mean(luminance < 0.05))
        message = f"dark(<0.05 luminance) fraction={dark_fraction:.5f} (n={int(contributing.sum())})"
        self.assertLessEqual(dark_fraction, 0.005, message)

    def test_report_shape(self):
        report = self.report
        self.assertEqual(sorted(report["views_used"]), sorted(ROLES), report["views_used"])
        # Zero-tilt side cameras cannot see tops or bottoms (the thin plates'
        # top+bottom are 60% of their area), so judge coverage where the four
        # views CAN reach: side-facing texels (|n_z| < 0.5) must be projected.
        texels = self.role_data["front"]["texels"]
        side_facing = np.abs(texels["normals"][:, 2]) < 0.5
        projected_flat = self.projected_mask.reshape(-1)[texels["covered_index"]]
        side_fraction = float(np.mean(projected_flat[side_facing]))
        self.assertGreaterEqual(
            side_fraction, 0.9,
            f"side-facing projected fraction={side_fraction:.4f} (n={int(side_facing.sum())}); "
            f"overall projected_texel_fraction={report['projected_texel_fraction']}",
        )
        self.assertGreaterEqual(
            report["projected_texel_fraction"], 0.5, report["projected_texel_fraction"],
        )
        for role in ROLES:
            coverage = report["per_role_coverage"][role]
            # 2026-09-04 GPU-box incident read exactly 0.0 for every role.
            self.assertGreaterEqual(coverage, 0.10, f"{role}: per_role_coverage={coverage}")
            diagnostics = report["diagnostics"][role]
            self.assertGreater(diagnostics["n_facing_positive"], 0, f"{role}: n_facing_positive=0")
            self.assertGreater(diagnostics["n_weight_positive"], 0, f"{role}: n_weight_positive=0")


class GuideViewExplicitCameraConformanceTests(unittest.TestCase):
    """task9: mesh-guided view synthesis. `render-guide-views` renders CLAY
    images FROM the fixture mesh's own exact per-role cameras (`guides.json`);
    `project-views` reuses those cameras verbatim (`settings["cameras"]`)
    instead of re-deriving them from this run's own mesh bounds, so the frame
    is identity and the extra ("front-left") role is accepted."""

    GUIDE_ROLES = ("front", "back", "left", "right", "front-left")

    @classmethod
    def setUpClass(cls):
        require_blender_4x()
        cls.tmp_dir = Path(tempfile.mkdtemp(prefix="aismith-guide-views-conformance-"))
        fixture_glb = cls.tmp_dir / "fixture.glb"
        atlas_png = cls.tmp_dir / "atlas.png"
        meta_json = cls.tmp_dir / "fixture_meta.json"
        native.build_projection_fixture(fixture_glb, atlas_png, meta_json, atlas_size=512)

        guides_dir = cls.tmp_dir / "guides"
        cls.guides = native.render_guide_views(
            fixture_glb, guides_dir,
            roles=("front", "back", "left", "right"), extra_roles=("front-left",), size=512,
        )
        clay_views = {role: Path(info["clay"]) for role, info in cls.guides["roles"].items()}

        cls.out_glb = cls.tmp_dir / "out.glb"
        with patch.dict(os.environ, {"AISMITH_PROJECT_VIEWS_DEBUG": "1"}):
            cls.report = native.run_project_views(
                fixture_glb, cls.out_glb, clay_views,
                {"cycles_samples": 8, "cameras": cls.guides["roles"]},
            )
        cls.render_dir = cls.out_glb.parent / "project-views-render"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_explicit_camera_roles_used_and_frame_is_identity(self):
        report = self.report
        self.assertIn("front-left", report["views_used"])
        self.assertEqual(sorted(report["cameras_explicit"]), sorted(self.GUIDE_ROLES))
        for role in self.GUIDE_ROLES:
            with self.subTest(role=role):
                alignment = report["frame_alignment"][role]
                self.assertAlmostEqual(alignment["scale"], 1.0, places=6)
                self.assertAlmostEqual(alignment["dx"], 0.0, places=6)
                self.assertAlmostEqual(alignment["dy"], 0.0, places=6)

    def test_front_left_depth_agreement(self):
        role = "front-left"
        with np.load(self.render_dir / f"{role}_texels.npz") as npz:
            texels = {key: npz[key] for key in npz.files}
        depth_raw = np.load(self.render_dir / f"{role}_depth.npy")
        alpha = np.load(self.render_dir / f"{role}_alpha.npy")
        meta = json.loads((self.render_dir / f"{role}_meta.json").read_text(encoding="utf-8"))
        camera_distance, depth_bias = meta["camera_distance"], meta["depth_bias"]

        ix = np.clip(texels["render_ix"], 0, meta["image_width"] - 1)
        iy = np.clip(texels["render_iy"], 0, meta["image_height"] - 1)
        alpha_at = alpha[iy, ix]
        T = texels["in_render_bounds"] & (texels["facing"] > 0.5) & (alpha_at > 0.5)
        self.assertGreater(int(T.sum()), 0, f"{role}: empty T population")

        delta = (depth_raw[iy, ix] - camera_distance) - texels["depth"]
        frac_within = float(np.mean(np.abs(delta[T]) <= depth_bias))
        self.assertGreaterEqual(
            frac_within, 0.98, f"{role}: frac_within_bias={frac_within:.4f} (n={int(T.sum())})",
        )


class PerspectiveCameraConformanceTests(unittest.TestCase):
    """The source-pose chain lays a PHOTO's pixels back over the mesh, and a
    photo is a PERSPECTIVE view from an unknown camera -- not one of the
    zero-tilt ortho conditioning cameras. Renders the same asymmetric fixture
    from a pitched perspective camera, hands that exact camera back to
    `project-views` as an explicit perspective camera under the chain's own
    role name ("photo", which no azimuth table knows -- so this also pins the
    unknown-role guard's "an explicit camera is enough" behaviour), and
    asserts the same per-texel depth and colour agreement the ortho cases do.
    """

    RENDER_ROLE = "front"
    PROJECT_ROLE = "photo"
    FOV_X_DEGREES = 40.0
    PITCH_DEGREES = 12.0
    SIZE = 512

    @classmethod
    def setUpClass(cls):
        require_blender_4x()
        cls.tmp_dir = Path(tempfile.mkdtemp(prefix="aismith-perspective-conformance-"))
        fixture_glb = cls.tmp_dir / "fixture.glb"
        cls.atlas_png = cls.tmp_dir / "atlas.png"
        meta_json = cls.tmp_dir / "fixture_meta.json"
        cls.fixture_meta = native.build_projection_fixture(
            fixture_glb, cls.atlas_png, meta_json, atlas_size=512,
        )

        views_dir = cls.tmp_dir / "views"
        rendered = native.render_role_views(
            fixture_glb, views_dir, roles=(cls.RENDER_ROLE,), size=cls.SIZE,
            projection="perspective", fov_x_degrees=cls.FOV_X_DEGREES, pitch_degrees=cls.PITCH_DEGREES,
        )
        cls.camera = json.loads((views_dir / "views.json").read_text(encoding="utf-8"))[cls.RENDER_ROLE]

        cls.out_glb = cls.tmp_dir / "out.glb"
        with patch.dict(os.environ, {"AISMITH_PROJECT_VIEWS_DEBUG": "1"}):
            cls.report = native.run_project_views(
                fixture_glb, cls.out_glb,
                {cls.PROJECT_ROLE: rendered[cls.RENDER_ROLE]},
                {"cycles_samples": 8, "cameras": {cls.PROJECT_ROLE: cls.camera}},
            )
        cls.render_dir = cls.out_glb.parent / "project-views-render"

        with np.load(cls.render_dir / f"{cls.PROJECT_ROLE}_texels.npz") as npz:
            cls.texels = {key: npz[key] for key in npz.files}
        cls.depth_raw = np.load(cls.render_dir / f"{cls.PROJECT_ROLE}_depth.npy")
        cls.alpha = np.load(cls.render_dir / f"{cls.PROJECT_ROLE}_alpha.npy")
        cls.meta = json.loads(
            (cls.render_dir / f"{cls.PROJECT_ROLE}_meta.json").read_text(encoding="utf-8")
        )
        cls.projected_mask = np.load(cls.render_dir / "projected_mask.npy")
        cls.atlas_out = np.load(cls.render_dir / "atlas_out.npy")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def _render_indices(self):
        return (
            np.clip(self.texels["render_ix"], 0, self.meta["image_width"] - 1),
            np.clip(self.texels["render_iy"], 0, self.meta["image_height"] - 1),
        )

    def _facing_bound_alpha_hit(self):
        """Same T population as the ortho cases: covered texels facing the
        camera, inside its render, whose render pixel hit geometry."""
        ix, iy = self._render_indices()
        alpha_at = self.alpha[iy, ix]
        return ix, iy, self.texels["in_render_bounds"] & (self.texels["facing"] > 0.5) & (alpha_at > 0.5)

    def _at_depth_discontinuity(self, ix, iy):
        """Does the render pixel's 3x3 neighbourhood span more than
        `depth_bias`? The one-sided neighbourhood-minimum test rejects that
        band by design (see the ortho case's own helper)."""
        valid = self.alpha > 0.5
        padded_lo = np.pad(np.where(valid, self.depth_raw, np.inf), 1, mode="edge")
        padded_hi = np.pad(np.where(valid, self.depth_raw, -np.inf), 1, mode="edge")
        h, w = self.depth_raw.shape
        windows_lo = np.stack([padded_lo[i:i + h, j:j + w] for i in range(3) for j in range(3)])
        windows_hi = np.stack([padded_hi[i:i + h, j:j + w] for i in range(3) for j in range(3)])
        span = windows_hi.max(axis=0) - windows_lo.min(axis=0)
        return span[iy, ix] > self.meta["depth_bias"]

    def test_camera_is_perspective_and_the_frame_is_identity(self):
        self.assertEqual(self.meta["kind"], "perspective")
        self.assertAlmostEqual(self.meta["fov_x_degrees"], self.FOV_X_DEGREES, places=6)
        self.assertAlmostEqual(self.meta["pitch_degrees"], self.PITCH_DEGREES, places=6)
        self.assertEqual(self.report["cameras_explicit"], [self.PROJECT_ROLE])
        self.assertEqual(self.report["views_used"], [self.PROJECT_ROLE])
        alignment = self.report["frame_alignment"][self.PROJECT_ROLE]
        self.assertAlmostEqual(alignment["scale"], 1.0, places=6)
        self.assertAlmostEqual(alignment["dx"], 0.0, places=6)
        self.assertAlmostEqual(alignment["dy"], 0.0, places=6)

    def test_depth_map_agrees_with_analytic_texel_depth(self):
        ix, iy, T = self._facing_bound_alpha_hit()
        camera_distance = self.meta["camera_distance"]
        diagonal, depth_bias = self.meta["diagonal"], self.meta["depth_bias"]
        self.assertGreater(int(T.sum()), 0, "empty T population")

        depth = self.texels["depth"]
        # Blender's Z pass for a perspective camera is the distance along the
        # view axis, the same frame the ortho path uses -- so the SAME
        # conversion applies. A radial-distance reading of the pass, or a
        # mirrored/mis-pitched basis, breaks exactly here.
        delta = (self.depth_raw[iy, ix] - camera_distance) - depth
        median_abs = float(np.median(np.abs(delta[T])))
        frac_within = float(np.mean(np.abs(delta[T]) <= depth_bias))
        smooth = T & ~self._at_depth_discontinuity(ix, iy)
        frac_within_smooth = float(np.mean(np.abs(delta[smooth]) <= depth_bias))

        # Diagnostics computed on failure's behalf, as in the ortho case: a
        # row-order bug and a sign bug each leave their own fingerprint, and
        # so does the one convention that only differs for a PERSPECTIVE
        # camera -- if Blender's Z pass were the RADIAL distance to the
        # camera rather than the distance along its view axis, the residual
        # below would be the one that fits.
        delta_flipped = (np.flipud(self.depth_raw)[iy, ix] - camera_distance) - depth
        basis = view_projection.tilted_view_basis(
            self.PROJECT_ROLE, self.meta["yaw_degrees"], self.meta["pitch_degrees"],
        )
        location = np.asarray(self.meta["center"], dtype=np.float64) + basis.direction * camera_distance
        radial = np.linalg.norm(self.texels["positions"] - location, axis=-1)
        delta_radial = self.depth_raw[iy, ix] - radial
        message = (
            f"median|delta|={median_abs:.6f} (tol {1e-3 * diagonal:.6f}), "
            f"frac_within_bias={frac_within:.4f} (n={int(T.sum())}, bias={depth_bias:.6f}), "
            f"away from discontinuities={frac_within_smooth:.4f} (n={int(smooth.sum())}) | "
            f"diag flipud: median|delta|={float(np.median(np.abs(delta_flipped[T]))):.6f}, "
            f"frac_within={float(np.mean(np.abs(delta_flipped[T]) <= depth_bias)):.4f} | "
            f"diag -delta: frac_within={float(np.mean(np.abs(-delta[T]) <= depth_bias)):.4f} | "
            f"diag radial-distance model: median|delta|={float(np.median(np.abs(delta_radial[T]))):.6f}, "
            f"frac_within={float(np.mean(np.abs(delta_radial[T]) <= depth_bias)):.4f}"
        )
        self.assertLessEqual(median_abs, 1e-3 * diagonal, message)
        self.assertGreaterEqual(frac_within, 0.98, message)
        self.assertGreaterEqual(frac_within_smooth, 0.99, message)

    def test_camera_facing_texels_pass_the_depth_test(self):
        sample_ix = np.round(self.texels["sample_px"]).astype(np.int64)
        sample_iy = np.round(self.texels["sample_py"]).astype(np.int64)
        in_frame = (
            (sample_ix >= 0) & (sample_ix < self.meta["image_width"])
            & (sample_iy >= 0) & (sample_iy < self.meta["image_height"])
        )
        F = self.texels["in_render_bounds"] & in_frame & (self.texels["facing"] > 0.5)
        self.assertGreater(int(F.sum()), 0, "empty F population")
        pass_rate = float(np.mean(self.texels["within_depth"][F]))
        self.assertGreaterEqual(pass_rate, 0.95, f"pass-rate={pass_rate:.4f} (n={int(F.sum())})")

    def test_projected_colours_match_the_fixture_atlas(self):
        # Same ground truth and the same interior-only judgement as the ortho
        # colour case: the fixture is self-emissive under a black world, so a
        # rendered pixel IS its atlas colour, and a mirrored or mis-projected
        # perspective camera sends whole faces to the wrong colour here.
        fixture_image = Image.open(self.atlas_png).convert("RGB")
        atlas_fixture = np.flipud(np.asarray(fixture_image, dtype=np.float64) / 255.0)
        mask = self.projected_mask
        self.assertGreater(int(mask.sum()), 0, "projected_mask selects no texels")

        palette = np.array(list(self.fixture_meta["face_colours"].values()), dtype=np.float64)

        def classify(values):
            return np.argmin(np.linalg.norm(values[:, None, :] - palette[None, :, :], axis=-1), axis=1)

        well_seen = np.zeros(mask.size, dtype=bool)
        well_seen[self.texels["covered_index"]] = self.texels["facing"] >= 0.5
        well_seen = well_seen.reshape(mask.shape) & mask

        label_map = classify(atlas_fixture.reshape(-1, 3)).reshape(atlas_fixture.shape[:2])
        radius = 8
        padded = np.pad(label_map, radius, mode="edge")
        interior = np.ones(label_map.shape, dtype=bool)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                shifted = padded[
                    radius + dy:radius + dy + label_map.shape[0],
                    radius + dx:radius + dx + label_map.shape[1],
                ]
                interior &= shifted == label_map
        judged = well_seen & interior
        out, expected = self.atlas_out[judged], atlas_fixture[judged]
        accuracy = float(np.mean(classify(out) == classify(expected)))
        median = float(np.median(np.abs(out - expected)))
        message = (
            f"well-seen interior texels: classification accuracy={accuracy:.4f}, "
            f"median|diff|={median:.4f} (n={int(judged.sum())}); "
            f"projected texels n={int(mask.sum())}"
        )
        # One view, so far fewer judged texels than the four-view ortho case.
        self.assertGreater(int(judged.sum()), 2000, message)
        self.assertGreaterEqual(accuracy, 0.99, message)
        self.assertLessEqual(median, 0.05, message)

    def test_report_shape(self):
        coverage = self.report["per_role_coverage"][self.PROJECT_ROLE]
        diagnostics = self.report["diagnostics"][self.PROJECT_ROLE]
        self.assertGreaterEqual(coverage, 0.10, f"per_role_coverage={coverage}")
        self.assertGreater(diagnostics["n_facing_positive"], 0)
        self.assertGreater(diagnostics["n_weight_positive"], 0)


if __name__ == "__main__":
    unittest.main()
