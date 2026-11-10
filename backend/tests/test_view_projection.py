import math
import unittest

import numpy as np

from backend.app import view_projection as vp
from backend.app.source_fidelity import SourceCamera


class OrthoViewBasisTests(unittest.TestCase):
    def test_world_z_maps_to_image_up_for_every_role(self):
        for role in vp.ROLE_AZIMUTH_DEGREES:
            basis = vp.ortho_view_basis(role)
            np.testing.assert_allclose(basis.up, (0.0, 0.0, 1.0))

    def test_front_camera_forward_is_plus_y(self):
        basis = vp.ortho_view_basis("front")
        np.testing.assert_allclose(basis.direction, (0.0, -1.0, 0.0), atol=1e-9)
        np.testing.assert_allclose(basis.forward, (0.0, 1.0, 0.0), atol=1e-9)

    def test_back_camera_forward_is_minus_y(self):
        basis = vp.ortho_view_basis("back")
        np.testing.assert_allclose(basis.forward, (0.0, -1.0, 0.0), atol=1e-9)

    def test_basis_vectors_are_unit_and_right_handed(self):
        for role in vp.ROLE_AZIMUTH_DEGREES:
            basis = vp.ortho_view_basis(role)
            for vector in (basis.right, basis.up, basis.forward):
                self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=9)
            # forward = -direction, and right = cross(forward, up) is
            # Blender's camera X axis (Y_cam x Z_cam with Z_cam = -forward).
            np.testing.assert_allclose(basis.forward, -basis.direction, atol=1e-9)
            np.testing.assert_allclose(basis.right, np.cross(basis.forward, basis.up), atol=1e-9)

    def test_horizontal_handedness_matches_blender_camera(self):
        # Front camera sits at -Y looking +Y with Z up: a viewer standing
        # there has world +X on their RIGHT, so +X must land at a larger
        # column. Left camera (at +X looking -X) has world +Y on its right.
        # The reversed cross product mirrors every view (conformance fixture,
        # 2026-09-05: side views 0.67 -> 0.99 within-bias once un-mirrored).
        for role, plus_axis in (("front", [1.0, 0.0, 0.0]), ("left", [0.0, 1.0, 0.0]),
                                ("back", [-1.0, 0.0, 0.0]), ("right", [0.0, -1.0, 0.0])):
            basis = vp.ortho_view_basis(role)
            px, _, _ = vp.project_points(
                np.array([[0.0, 0.0, 0.0], plus_axis]), basis, center=np.zeros(3), ortho_scale=2.0, image_size=100,
            )
            self.assertGreater(float(px[1]), float(px[0]), role)

    def test_unknown_role_raises(self):
        with self.assertRaises(ValueError):
            vp.ortho_view_basis("top")


class AzimuthViewBasisTests(unittest.TestCase):
    def test_arbitrary_azimuth_matches_the_same_maths_as_the_table(self):
        basis = vp.azimuth_view_basis("front-left", 45.0)
        angle = math.radians(45.0)
        np.testing.assert_allclose(
            basis.direction, (math.sin(angle), -math.cos(angle), 0.0), atol=1e-9,
        )
        self.assertAlmostEqual(float(np.linalg.norm(basis.right)), 1.0, places=9)
        np.testing.assert_allclose(basis.right, np.cross(basis.forward, basis.up), atol=1e-9)

    def test_ortho_view_basis_is_a_thin_wrapper_over_the_table(self):
        table_basis = vp.ortho_view_basis("left")
        explicit_basis = vp.azimuth_view_basis("left", 90.0)
        self.assertEqual(table_basis.role, explicit_basis.role)
        np.testing.assert_allclose(table_basis.direction, explicit_basis.direction, atol=1e-12)
        np.testing.assert_allclose(table_basis.right, explicit_basis.right, atol=1e-12)
        np.testing.assert_allclose(table_basis.up, explicit_basis.up, atol=1e-12)
        np.testing.assert_allclose(table_basis.forward, explicit_basis.forward, atol=1e-12)


class ResolveRoleAzimuthTests(unittest.TestCase):
    def test_table_role_with_no_cameras(self):
        self.assertEqual(vp.resolve_role_azimuth("front", None), 0.0)
        self.assertEqual(vp.resolve_role_azimuth("left", {}), 90.0)

    def test_extra_role_with_no_cameras(self):
        self.assertEqual(vp.resolve_role_azimuth("front-left", None), 45.0)
        self.assertEqual(vp.resolve_role_azimuth("back-right", {}), -135.0)

    def test_explicit_camera_overrides_the_table(self):
        cameras = {"front": {"azimuth_degrees": 12.0}}
        self.assertEqual(vp.resolve_role_azimuth("front", cameras), 12.0)

    def test_explicit_camera_covers_an_unknown_role(self):
        cameras = {"three-quarter-up": {"azimuth_degrees": 20.0}}
        self.assertEqual(vp.resolve_role_azimuth("three-quarter-up", cameras), 20.0)

    def test_unknown_role_without_camera_raises(self):
        with self.assertRaises(ValueError):
            vp.resolve_role_azimuth("top", None)
        with self.assertRaises(ValueError):
            vp.resolve_role_azimuth("top", {"front": {"azimuth_degrees": 0.0}})


class ProjectPointsTests(unittest.TestCase):
    def test_unit_cube_pixel_extents_front(self):
        basis = vp.ortho_view_basis("front")
        corners = np.array(
            [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
        )
        px, py, depth = vp.project_points(corners, basis, center=np.zeros(3), ortho_scale=1.0, image_size=100)
        self.assertAlmostEqual(float(px.min()), 0.0, places=6)
        self.assertAlmostEqual(float(px.max()), 100.0, places=6)
        self.assertAlmostEqual(float(py.min()), 0.0, places=6)
        self.assertAlmostEqual(float(py.max()), 100.0, places=6)
        # y = -0.5 is closest to the front camera (depth 0 at y=0.5 world
        # units away is farther); depth increases moving away from camera.
        self.assertLess(depth.min(), depth.max())

    def test_center_projects_to_image_center(self):
        basis = vp.ortho_view_basis("left")
        px, py, depth = vp.project_points(np.zeros((1, 3)), basis, center=np.zeros(3), ortho_scale=2.0, image_size=64)
        self.assertAlmostEqual(float(px[0]), 32.0)
        self.assertAlmostEqual(float(py[0]), 32.0)
        self.assertAlmostEqual(float(depth[0]), 0.0)

    def test_plus_z_point_maps_to_a_larger_row_index(self):
        # Bottom-up: row 0 = v = 0 = the bottom of the image, matching
        # Blender's own `image.pixels` buffer order, so +Z (up) moves to a
        # LARGER row index, not a smaller one.
        basis = vp.ortho_view_basis("right")
        px, py, _ = vp.project_points(
            np.array([[0.0, 0.0, 0.5]]), basis, center=np.zeros(3), ortho_scale=1.0, image_size=100
        )
        self.assertGreater(float(py[0]), 50.0)


class FitFrameToSubjectTests(unittest.TestCase):
    def test_identity_when_bboxes_match(self):
        bbox = (10.0, 20.0, 90.0, 180.0)
        scale, dx, dy = vp.fit_frame_to_subject(bbox, bbox)
        self.assertAlmostEqual(scale, 1.0)
        self.assertAlmostEqual(dx, 0.0)
        self.assertAlmostEqual(dy, 0.0)
        px, py = vp.apply_frame(np.array([10.0, 90.0]), np.array([20.0, 180.0]), scale, dx, dy)
        np.testing.assert_allclose(px, [10.0, 90.0])
        np.testing.assert_allclose(py, [20.0, 180.0])

    def test_scales_and_centers_mesh_bbox_onto_image_bbox(self):
        mesh_bbox = (0.0, 0.0, 50.0, 50.0)
        image_bbox = (10.0, 10.0, 110.0, 110.0)
        scale, dx, dy = vp.fit_frame_to_subject(mesh_bbox, image_bbox)
        self.assertAlmostEqual(scale, 2.0)
        px, py = vp.apply_frame(np.array([0.0, 25.0, 50.0]), np.array([0.0, 25.0, 50.0]), scale, dx, dy)
        np.testing.assert_allclose(px, [10.0, 60.0, 110.0])
        np.testing.assert_allclose(py, [10.0, 60.0, 110.0])

    def test_degenerate_mesh_bbox_raises(self):
        with self.assertRaises(ValueError):
            vp.fit_frame_to_subject((5.0, 5.0, 5.0, 5.0), (0.0, 0.0, 10.0, 10.0))


class InvertFrameTests(unittest.TestCase):
    def test_round_trips_apply_frame(self):
        scale, dx, dy = 1.4, -8.0, 12.5
        px = np.array([0.0, 10.0, 55.0])
        py = np.array([0.0, 20.0, 90.0])
        framed_px, framed_py = vp.apply_frame(px, py, scale, dx, dy)
        inv_scale, inv_dx, inv_dy = vp.invert_frame(scale, dx, dy)
        back_px, back_py = vp.apply_frame(framed_px, framed_py, inv_scale, inv_dx, inv_dy)
        np.testing.assert_allclose(back_px, px, atol=1e-9)
        np.testing.assert_allclose(back_py, py, atol=1e-9)

    def test_identity_frame_inverts_to_itself(self):
        self.assertEqual(vp.invert_frame(1.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    def test_zero_scale_raises(self):
        with self.assertRaises(ValueError):
            vp.invert_frame(0.0, 1.0, 2.0)


class MinFilterNeighborhoodTests(unittest.TestCase):
    def test_isolated_low_value_spreads_across_the_neighborhood(self):
        array = np.full((5, 5), 10.0)
        array[2, 2] = 1.0
        filtered = vp.min_filter_neighborhood(array, radius=1)
        # Every texel within a 3x3 neighborhood of (2, 2) now reads 1.0.
        self.assertTrue(np.all(filtered[1:4, 1:4] == 1.0))
        self.assertEqual(filtered[0, 0], 10.0)

    def test_radius_zero_is_a_no_op(self):
        array = np.array([[1.0, 5.0], [9.0, 2.0]])
        np.testing.assert_array_equal(vp.min_filter_neighborhood(array, radius=0), array)

    def test_edges_are_clamped_not_padded_with_a_phantom_minimum(self):
        array = np.array([[5.0, 5.0], [5.0, 5.0]])
        filtered = vp.min_filter_neighborhood(array, radius=1)
        np.testing.assert_array_equal(filtered, array)


class ErodeMaskTests(unittest.TestCase):
    def test_shrinks_a_filled_square_by_radius(self):
        mask = np.zeros((11, 11), dtype=bool)
        mask[2:9, 2:9] = True  # a 7x7 filled square
        eroded = vp.erode_mask(mask, radius=1)
        expected = np.zeros((11, 11), dtype=bool)
        expected[3:8, 3:8] = True
        np.testing.assert_array_equal(eroded, expected)

    def test_a_region_touching_the_border_shrinks_from_that_edge_too(self):
        mask = np.ones((7, 7), dtype=bool)
        eroded = vp.erode_mask(mask, radius=1)
        expected = np.zeros((7, 7), dtype=bool)
        expected[1:6, 1:6] = True
        np.testing.assert_array_equal(eroded, expected)

    def test_radius_zero_is_a_no_op(self):
        mask = np.array([[True, False], [False, True]])
        np.testing.assert_array_equal(vp.erode_mask(mask, radius=0), mask)


class MaskBoundaryRingTests(unittest.TestCase):
    def test_ring_is_true_only_near_the_boundary(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:18, 2:18] = True  # a 16x16 filled square, boundary at row/col 2 and 17
        ring = vp.mask_boundary_ring(mask, radius=4)
        # Just inside the boundary: within 4px of row/col 2 or 17.
        self.assertTrue(ring[3, 3])
        self.assertTrue(ring[16, 16])
        # Deep interior: more than 4px from every edge.
        self.assertFalse(ring[10, 10])
        # Outside the mask entirely.
        self.assertFalse(ring[0, 0])

    def test_a_mask_no_wider_than_twice_the_radius_is_all_ring(self):
        mask = np.ones((6, 6), dtype=bool)
        ring = vp.mask_boundary_ring(mask, radius=4)
        np.testing.assert_array_equal(ring, mask)


class BackgroundBleedMaskTests(unittest.TestCase):
    """Pins the 2026-09-04 GPU-box regression: a global background-colour
    guard rejected a pale-skinned alien's face because its skin sat close
    in luminance to the light-gray conditioning background. The guard must
    only ever fire within a ring near the (already-eroded) mask's own
    boundary, never in the deep interior -- regardless of how close an
    interior colour happens to sit to the background.
    """

    def _grid(self, size: int):
        rows, cols = np.mgrid[0:size, 0:size]
        return cols.astype(np.float64), rows.astype(np.float64)

    def test_pale_interior_skin_is_accepted_background_matching_ring_is_rejected(self):
        size = 20
        subject_mask = np.zeros((size, size), dtype=bool)
        subject_mask[2:18, 2:18] = True
        background_color = np.array([0.75, 0.75, 0.75])
        sample_px, sample_py = self._grid(size)

        colors = np.full((size, size, 3), 0.9)  # pale skin, everywhere
        bleed = vp.background_bleed_mask(colors, subject_mask, background_color, sample_px, sample_py)
        self.assertFalse(bleed[10, 10], "deep interior pale skin must never be rejected")

        colors_ring_bleed = np.full((size, size, 3), 0.75)  # background-colored bleed
        bleed_ring = vp.background_bleed_mask(
            colors_ring_bleed, subject_mask, background_color, sample_px, sample_py,
        )
        self.assertTrue(bleed_ring[3, 3], "a ring texel matching the background must be rejected")
        self.assertFalse(bleed_ring[10, 10], "the deep interior is never touched, even background-colored")

    def test_interior_pixel_numerically_matching_background_is_still_accepted(self):
        # The case that actually distinguishes a ring-scoped guard from a
        # global one: an interior sample that happens to equal the
        # background colour exactly must still be kept.
        size = 20
        subject_mask = np.zeros((size, size), dtype=bool)
        subject_mask[2:18, 2:18] = True
        background_color = np.array([0.75, 0.75, 0.75])
        sample_px, sample_py = self._grid(size)
        colors = np.full((size, size, 3), 0.75)

        bleed = vp.background_bleed_mask(colors, subject_mask, background_color, sample_px, sample_py)
        self.assertFalse(bleed[10, 10])
        self.assertTrue(bleed[3, 3])

    def test_outside_the_mask_is_never_flagged(self):
        size = 20
        subject_mask = np.zeros((size, size), dtype=bool)
        subject_mask[2:18, 2:18] = True
        background_color = np.array([0.75, 0.75, 0.75])
        sample_px, sample_py = self._grid(size)
        colors = np.full((size, size, 3), 0.75)

        bleed = vp.background_bleed_mask(colors, subject_mask, background_color, sample_px, sample_py)
        self.assertFalse(bleed[0, 0])


class RasterizeAtlasTests(unittest.TestCase):
    def test_two_triangles_cover_whole_atlas(self):
        uv_triangles = np.array(
            [
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            ]
        )
        world_triangles = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            ]
        )
        normal_triangles = np.zeros_like(world_triangles)
        normal_triangles[..., 2] = 1.0

        positions, normals, covered = vp.rasterize_atlas(uv_triangles, world_triangles, normal_triangles, atlas_size=64)
        self.assertTrue(covered.all())
        selected_normals = normals[covered]
        np.testing.assert_allclose(
            selected_normals, np.broadcast_to([0.0, 0.0, 1.0], selected_normals.shape), atol=1e-6
        )

        # world position equals (u, v, 0) by construction, and atlas row 0 is
        # v = 0.0 (bottom-up, matching Blender's own image.pixels buffer
        # order) -- so array index [0, 0] (row 0, col 0) is near (0, 0, 0)
        # and [-1, -1] (last row, last col) is near (1, 1, 0).
        bottom_left = positions[0, 0]
        top_right = positions[-1, -1]
        np.testing.assert_allclose(bottom_left, [0.0, 0.0, 0.0], atol=0.05)
        np.testing.assert_allclose(top_right, [1.0, 1.0, 0.0], atol=0.05)

        # Linear interpolation: the atlas center should sit near (0.5, 0.5, 0).
        center = positions[32, 32]
        np.testing.assert_allclose(center, [0.5, 0.5, 0.0], atol=0.05)

    def test_degenerate_uv_triangle_is_skipped_without_error(self):
        uv_triangles = np.array([[[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]])
        world_triangles = np.zeros((1, 3, 3))
        normal_triangles = np.zeros((1, 3, 3))
        positions, normals, covered = vp.rasterize_atlas(uv_triangles, world_triangles, normal_triangles, atlas_size=16)
        self.assertFalse(covered.any())


class FacingSignConventionTests(unittest.TestCase):
    """Pins the facing-sign convention this module's docstring states:
    weight rewards `dot(normal, basis.direction)`, i.e. a texel whose normal
    points TOWARD the camera -- not `dot(normal, basis.forward)`, which
    would instead reward the mesh's far side. Named after the exact scenario
    the "front" role describes: camera at -Y (direction=(0,-1,0)) looking
    toward +Y, so the mesh's -Y-facing surface (normal near (0,-1,0)) is
    what that camera actually sees.
    """

    def _weight_ignoring_depth_and_mask(self, basis, normal):
        image_size = 8
        center = np.zeros(3)
        ortho_scale = 2.0
        positions = np.zeros((1, 1, 3))
        normals = np.zeros((1, 1, 3))
        normals[0, 0] = normal
        depth_map = np.zeros((image_size, image_size))
        subject_mask = np.ones((image_size, image_size), dtype=bool)
        weights, _, _ = vp.view_texel_weights(
            positions, normals, basis, center, ortho_scale, image_size,
            depth_map, depth_bias=float("inf"), subject_mask=subject_mask,
            frame=(1.0, 0.0, 0.0), camera_distance=0.0,
        )
        return float(weights[0, 0])

    def test_minus_y_face_faces_the_front_camera(self):
        basis = vp.ortho_view_basis("front")
        weight = self._weight_ignoring_depth_and_mask(basis, normal=(0.0, -1.0, 0.0))
        self.assertAlmostEqual(weight, 1.0, places=9)

    def test_plus_y_face_faces_away_from_the_front_camera(self):
        basis = vp.ortho_view_basis("front")
        weight = self._weight_ignoring_depth_and_mask(basis, normal=(0.0, 1.0, 0.0))
        self.assertEqual(weight, 0.0)

    def test_every_role_rewards_its_own_direction_not_its_forward(self):
        for role in vp.ROLE_AZIMUTH_DEGREES:
            basis = vp.ortho_view_basis(role)
            toward_camera_weight = self._weight_ignoring_depth_and_mask(basis, normal=tuple(basis.direction))
            away_from_camera_weight = self._weight_ignoring_depth_and_mask(basis, normal=tuple(basis.forward))
            self.assertAlmostEqual(toward_camera_weight, 1.0, places=9, msg=role)
            self.assertEqual(away_from_camera_weight, 0.0, msg=role)


class ViewTexelWeightsTests(unittest.TestCase):
    def _setup(self, image_size=32):
        basis = vp.ortho_view_basis("front")
        center = np.zeros(3)
        ortho_scale = 2.0
        depth_map = np.full((image_size, image_size), 5.0)
        subject_mask = np.ones((image_size, image_size), dtype=bool)
        frame = (1.0, 0.0, 0.0)
        return basis, center, ortho_scale, image_size, depth_map, subject_mask, frame

    def test_facing_texel_at_matching_depth_gets_weight_near_one(self):
        basis, center, ortho_scale, image_size, depth_map, subject_mask, frame = self._setup()
        positions = np.zeros((1, 1, 3))
        normals = np.zeros((1, 1, 3))
        normals[0, 0] = basis.direction  # normal faces the camera exactly
        px, py, depth = vp.project_points(positions.reshape(-1, 3), basis, center, ortho_scale, image_size)
        depth_map[int(round(py[0])), int(round(px[0]))] = float(depth[0])

        weights, sample_px, sample_py = vp.view_texel_weights(
            positions, normals, basis, center, ortho_scale, image_size,
            depth_map, depth_bias=0.01, subject_mask=subject_mask, frame=frame, camera_distance=0.0,
        )
        self.assertAlmostEqual(float(weights[0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(sample_px[0, 0]), float(px[0]))
        self.assertAlmostEqual(float(sample_py[0, 0]), float(py[0]))

    def test_camera_distance_shifts_a_raw_blender_depth_pass_into_agreement(self):
        # Blender's Z/Depth pass is the ABSOLUTE distance from the camera,
        # not from `center` -- this is the exact bug the 2026-09-04 GPU-box
        # incident traced to. With the one-sided "texel_depth <= neighborhood
        # + bias" visibility test, forgetting to convert it manifests as a
        # FALSE ACCEPT (an unconverted, much-too-large reference value makes
        # any real texel look "in front of" it), not a false reject -- an
        # occluded texel (10 world units behind the true visible surface,
        # which sits at depth_from_center = 0) must be correctly rejected
        # once the camera's actual distance from `center` is supplied.
        basis, center, ortho_scale, image_size, _depth_map, subject_mask, frame = self._setup()
        px, py, _ = vp.project_points(np.zeros((1, 3)), basis, center, ortho_scale, image_size)
        camera_distance = 37.5

        positions = np.zeros((1, 1, 3))
        positions[0, 0] = np.asarray(basis.forward) * 10.0
        normals = np.zeros((1, 1, 3))
        normals[0, 0] = basis.direction
        # Blender's absolute Z at that pixel says the visible surface there
        # sits at depth_from_center = 0, i.e. absolute = 0 + camera_distance.
        depth_map_absolute = np.full((image_size, image_size), 9999.0)
        depth_map_absolute[int(round(py[0])), int(round(px[0]))] = camera_distance

        falsely_accepted, _, _ = vp.view_texel_weights(
            positions, normals, basis, center, ortho_scale, image_size,
            depth_map_absolute, depth_bias=0.01, subject_mask=subject_mask, frame=frame, camera_distance=0.0,
        )
        self.assertEqual(float(falsely_accepted[0, 0]), 1.0)

        correctly_rejected, _, _ = vp.view_texel_weights(
            positions, normals, basis, center, ortho_scale, image_size,
            depth_map_absolute, depth_bias=0.01, subject_mask=subject_mask, frame=frame,
            camera_distance=camera_distance,
        )
        self.assertEqual(float(correctly_rejected[0, 0]), 0.0)

    def test_normal_facing_away_gets_zero(self):
        basis, center, ortho_scale, image_size, depth_map, subject_mask, frame = self._setup()
        positions = np.zeros((1, 1, 3))
        normals = np.zeros((1, 1, 3))
        normals[0, 0] = basis.forward  # points away from the camera
        depth_map[:, :] = 0.0

        weights, _, _ = vp.view_texel_weights(
            positions, normals, basis, center, ortho_scale, image_size,
            depth_map, depth_bias=100.0, subject_mask=subject_mask, frame=frame, camera_distance=0.0,
        )
        self.assertEqual(float(weights[0, 0]), 0.0)

    def test_texel_behind_depth_surface_gets_zero(self):
        basis, center, ortho_scale, image_size, depth_map, subject_mask, frame = self._setup()
        positions = np.zeros((1, 1, 3))
        normals = np.zeros((1, 1, 3))
        normals[0, 0] = basis.direction
        px, py, depth = vp.project_points(positions.reshape(-1, 3), basis, center, ortho_scale, image_size)
        # Depth map says the visible surface is much closer to the camera
        # than this texel, i.e. this texel is occluded behind it.
        depth_map[int(round(py[0])), int(round(px[0]))] = float(depth[0]) - 10.0

        weights, _, _ = vp.view_texel_weights(
            positions, normals, basis, center, ortho_scale, image_size,
            depth_map, depth_bias=0.01, subject_mask=subject_mask, frame=frame, camera_distance=0.0,
        )
        self.assertEqual(float(weights[0, 0]), 0.0)

    def test_texel_outside_subject_mask_gets_zero(self):
        basis, center, ortho_scale, image_size, depth_map, subject_mask, frame = self._setup()
        positions = np.zeros((1, 1, 3))
        normals = np.zeros((1, 1, 3))
        normals[0, 0] = basis.direction
        px, py, depth = vp.project_points(positions.reshape(-1, 3), basis, center, ortho_scale, image_size)
        depth_map[int(round(py[0])), int(round(px[0]))] = float(depth[0])
        subject_mask[:, :] = False

        weights, _, _ = vp.view_texel_weights(
            positions, normals, basis, center, ortho_scale, image_size,
            depth_map, depth_bias=0.01, subject_mask=subject_mask, frame=frame, camera_distance=0.0,
        )
        self.assertEqual(float(weights[0, 0]), 0.0)

    def test_depth_map_shape_mismatch_raises_a_clear_error(self):
        # The live GPU-box failure this guards: a depth map (e.g. a stale
        # compositor buffer) that does not match image_size must fail loudly
        # rather than silently sampling the wrong texel's depth.
        basis, center, ortho_scale, image_size, _depth_map, subject_mask, frame = self._setup()
        wrong_size_depth_map = np.full((256, 256), 5.0)
        positions = np.zeros((1, 1, 3))
        normals = np.zeros((1, 1, 3))
        normals[0, 0] = basis.direction

        with self.assertRaisesRegex(ValueError, r"\(32, 32\).*\(256, 256\)"):
            vp.view_texel_weights(
                positions, normals, basis, center, ortho_scale, image_size,
                wrong_size_depth_map, depth_bias=0.01, subject_mask=subject_mask, frame=frame, camera_distance=0.0,
            )

    def test_diagnostics_funnel_pinpoints_where_a_texel_is_lost(self):
        # One facing, matching-depth, in-mask texel and one facing-away
        # texel, over a 1x2 "atlas" -- so the funnel counts are small enough
        # to assert on exactly.
        basis, center, ortho_scale, image_size, depth_map, subject_mask, frame = self._setup()
        positions = np.zeros((1, 2, 3))
        normals = np.zeros((1, 2, 3))
        normals[0, 0] = basis.direction
        normals[0, 1] = basis.forward
        px, py, depth = vp.project_points(positions[0, 0].reshape(1, 3), basis, center, ortho_scale, image_size)
        depth_map[int(round(py[0])), int(round(px[0]))] = float(depth[0])
        covered = np.array([[True, True]])

        diagnostics = vp.view_texel_diagnostics(
            positions, normals, basis, center, ortho_scale, image_size,
            depth_map, depth_bias=0.01, subject_mask=subject_mask, frame=frame,
            camera_distance=0.0, covered=covered,
        )
        self.assertEqual(diagnostics["n_covered"], 2)
        self.assertEqual(diagnostics["n_in_frame"], 2)
        self.assertEqual(diagnostics["n_depth_visible"], 2)
        self.assertEqual(diagnostics["n_in_subject_mask"], 2)
        self.assertEqual(diagnostics["n_facing_positive"], 1)
        self.assertEqual(diagnostics["n_weight_positive"], 1)

    def test_diagnostics_excludes_uncovered_texels(self):
        basis, center, ortho_scale, image_size, depth_map, subject_mask, frame = self._setup()
        positions = np.zeros((1, 1, 3))
        normals = np.zeros((1, 1, 3))
        normals[0, 0] = basis.direction
        covered = np.array([[False]])

        diagnostics = vp.view_texel_diagnostics(
            positions, normals, basis, center, ortho_scale, image_size,
            depth_map, depth_bias=0.01, subject_mask=subject_mask, frame=frame,
            camera_distance=0.0, covered=covered,
        )
        self.assertEqual(diagnostics["n_covered"], 0)
        self.assertEqual(diagnostics["n_weight_positive"], 0)

    def test_debug_masks_agree_with_the_diagnostics_funnel(self):
        # Same scenario as the funnel test above: debug_masks' per-texel
        # arrays must sum to exactly the same counts view_texel_diagnostics
        # reports, since both are views onto the same underlying computation.
        basis, center, ortho_scale, image_size, depth_map, subject_mask, frame = self._setup()
        positions = np.zeros((1, 2, 3))
        normals = np.zeros((1, 2, 3))
        normals[0, 0] = basis.direction
        normals[0, 1] = basis.forward
        px, py, depth = vp.project_points(positions[0, 0].reshape(1, 3), basis, center, ortho_scale, image_size)
        depth_map[int(round(py[0])), int(round(px[0]))] = float(depth[0])
        covered = np.array([[True, True]])

        debug_masks = vp.view_texel_debug_masks(
            positions, normals, basis, center, ortho_scale, image_size,
            depth_map, depth_bias=0.01, subject_mask=subject_mask, frame=frame,
            camera_distance=0.0,
        )
        self.assertEqual(debug_masks["in_frame"].shape, (1, 2))
        self.assertEqual(int((covered & debug_masks["in_frame"]).sum()), 2)
        self.assertEqual(int((covered & debug_masks["depth_visible"]).sum()), 2)
        self.assertEqual(int((covered & debug_masks["in_mask"]).sum()), 2)
        self.assertEqual(int((covered & debug_masks["facing_positive"]).sum()), 1)
        self.assertEqual(int((covered & (debug_masks["weights"] > 0.0)).sum()), 1)
        # texel 0 (normal toward the camera) is the one that faces and
        # carries positive weight; texel 1 (normal away) does neither.
        self.assertTrue(debug_masks["facing_positive"][0, 0])
        self.assertFalse(debug_masks["facing_positive"][0, 1])

    def test_debug_masks_expose_flat_per_texel_arrays(self):
        # The conformance fixture (task5) reads these directly instead of
        # re-deriving _view_texel_masks' math; pins their shape (flat, unlike
        # the original five (H, W) keys) and that depth_at_render is genuinely
        # the 3x3-min, not the exact render pixel.
        basis, center, ortho_scale, image_size, depth_map, subject_mask, frame = self._setup()
        positions = np.zeros((1, 2, 3))
        normals = np.zeros((1, 2, 3))
        normals[0, 0] = basis.direction
        normals[0, 1] = basis.forward
        px, py, depth = vp.project_points(positions[0, 0].reshape(1, 3), basis, center, ortho_scale, image_size)
        ix, iy = int(round(px[0])), int(round(py[0]))
        depth_map[iy, ix] = float(depth[0]) + 0.05
        depth_map[iy, ix - 1] = float(depth[0])  # lower neighbor the 3x3 min must pick up

        debug_masks = vp.view_texel_debug_masks(
            positions, normals, basis, center, ortho_scale, image_size,
            depth_map, depth_bias=0.2, subject_mask=subject_mask, frame=frame,
            camera_distance=0.0,
        )
        flat_keys = (
            "depth", "depth_at_render", "render_px", "render_py", "render_ix", "render_iy",
            "sample_px", "sample_py", "facing", "in_render_bounds", "within_depth",
        )
        for key in flat_keys:
            self.assertEqual(debug_masks[key].shape, (2,), key)

        expected_min = vp.min_filter_neighborhood(depth_map, radius=1)[iy, ix]
        self.assertAlmostEqual(float(debug_masks["depth_at_render"][0]), float(expected_min), places=9)
        self.assertNotAlmostEqual(float(debug_masks["depth_at_render"][0]), float(depth_map[iy, ix]), places=9)
        self.assertEqual(int(debug_masks["render_ix"][0]), ix)
        self.assertEqual(int(debug_masks["render_iy"][0]), iy)


class FlatQuadDepthConventionTests(unittest.TestCase):
    """Pins the depth reference-frame convention end to end (2026-09-05
    GPU-box follow-up): a flat quad facing the camera at a KNOWN world
    position and KNOWN camera_distance must read back texel_depth ==
    depth_map (both center-relative, after `camera_distance` conversion)
    to within 1e-6. This is the exact contract every depth comparison in
    this module relies on.
    """

    def test_flat_quad_at_center_matches_depth_map_exactly(self):
        basis = vp.ortho_view_basis("front")
        center = np.zeros(3)
        ortho_scale = 2.0
        image_size = 16
        camera_distance = 10.0

        positions = np.zeros((image_size, image_size, 3))
        normals = np.zeros((image_size, image_size, 3))
        normals[:, :] = basis.direction
        covered = np.ones((image_size, image_size), dtype=bool)
        subject_mask = np.ones((image_size, image_size), dtype=bool)
        frame = (1.0, 0.0, 0.0)

        # A flat quad AT `center` has depth_from_center = 0, so Blender's
        # own (camera-absolute) Z/Depth pass there is exactly camera_distance.
        depth_map = np.full((image_size, image_size), camera_distance)
        depth_map_valid = np.ones((image_size, image_size), dtype=bool)

        diagnostics = vp.view_texel_diagnostics(
            positions, normals, basis, center, ortho_scale, image_size,
            depth_map, depth_bias=1e-6, subject_mask=subject_mask, frame=frame,
            camera_distance=camera_distance, covered=covered,
            depth_map_valid=depth_map_valid,
        )
        self.assertAlmostEqual(diagnostics["texel_depth_min"], 0.0, places=6)
        self.assertAlmostEqual(diagnostics["texel_depth_max"], 0.0, places=6)
        self.assertAlmostEqual(diagnostics["depth_map_min"], 0.0, places=6)
        self.assertAlmostEqual(diagnostics["depth_map_max"], 0.0, places=6)
        self.assertEqual(diagnostics["n_depth_visible"], image_size * image_size)

    def test_flat_quad_offset_from_center_still_matches(self):
        # Same convention, but the quad sits away from `center` along the
        # camera's own view axis -- proves the conversion is not only
        # correct for the degenerate depth_from_center = 0 case.
        basis = vp.ortho_view_basis("front")
        center = np.zeros(3)
        ortho_scale = 2.0
        image_size = 16
        camera_distance = 10.0
        depth_from_center = 0.3

        positions = np.zeros((image_size, image_size, 3))
        positions[:, :] = np.asarray(basis.forward) * depth_from_center
        normals = np.zeros((image_size, image_size, 3))
        normals[:, :] = basis.direction
        covered = np.ones((image_size, image_size), dtype=bool)
        subject_mask = np.ones((image_size, image_size), dtype=bool)
        frame = (1.0, 0.0, 0.0)

        depth_map = np.full((image_size, image_size), depth_from_center + camera_distance)
        depth_map_valid = np.ones((image_size, image_size), dtype=bool)

        diagnostics = vp.view_texel_diagnostics(
            positions, normals, basis, center, ortho_scale, image_size,
            depth_map, depth_bias=1e-6, subject_mask=subject_mask, frame=frame,
            camera_distance=camera_distance, covered=covered,
            depth_map_valid=depth_map_valid,
        )
        self.assertAlmostEqual(diagnostics["texel_depth_min"], depth_from_center, places=6)
        self.assertAlmostEqual(diagnostics["depth_map_min"], depth_from_center, places=6)
        self.assertEqual(diagnostics["n_depth_visible"], image_size * image_size)


class DepthMapValidRegressionTests(unittest.TestCase):
    """Reproduces the 2026-09-05 GPU-box mechanism directly: a background
    pixel's depth sentinel (observed near 0, which becomes a huge NEGATIVE
    depth after the camera_distance shift) sitting immediately next to a
    real, visible surface pixel. Without `depth_map_valid`, the 3x3
    min-filter drags that fake near value in and the real pixel fails its
    own depth test even though nothing actually occludes it -- devastating
    for a thin part of the mesh (an arm, an ear), where almost every
    surface pixel is adjacent to a background pixel along its whole length.
    """

    def test_unmasked_background_falsely_occludes_an_adjacent_real_pixel(self):
        basis = vp.ortho_view_basis("front")
        center = np.zeros(3)
        ortho_scale = 2.0
        image_size = 5
        camera_distance = 100.0

        depth_map = np.full((image_size, image_size), camera_distance)
        depth_map[2, 3] = 0.0  # one background pixel, right next door

        positions = np.zeros((1, 1, 3))
        normals = np.zeros((1, 1, 3))
        normals[0, 0] = basis.direction
        subject_mask = np.ones((image_size, image_size), dtype=bool)
        frame = (1.0, 0.0, 0.0)

        px, py, _ = vp.project_points(positions.reshape(-1, 3), basis, center, ortho_scale, image_size)
        self.assertEqual((int(round(py[0])), int(round(px[0]))), (2, 2))

        without_masking, _, _ = vp.view_texel_weights(
            positions, normals, basis, center, ortho_scale, image_size,
            depth_map, depth_bias=0.01, subject_mask=subject_mask, frame=frame,
            camera_distance=camera_distance,
        )
        self.assertEqual(float(without_masking[0, 0]), 0.0)

    def test_depth_map_valid_excludes_the_background_sentinel(self):
        basis = vp.ortho_view_basis("front")
        center = np.zeros(3)
        ortho_scale = 2.0
        image_size = 5
        camera_distance = 100.0

        depth_map = np.full((image_size, image_size), camera_distance)
        depth_map[2, 3] = 0.0
        depth_map_valid = np.ones((image_size, image_size), dtype=bool)
        depth_map_valid[2, 3] = False

        positions = np.zeros((1, 1, 3))
        normals = np.zeros((1, 1, 3))
        normals[0, 0] = basis.direction
        subject_mask = np.ones((image_size, image_size), dtype=bool)
        frame = (1.0, 0.0, 0.0)

        with_masking, _, _ = vp.view_texel_weights(
            positions, normals, basis, center, ortho_scale, image_size,
            depth_map, depth_bias=0.01, subject_mask=subject_mask, frame=frame,
            camera_distance=camera_distance, depth_map_valid=depth_map_valid,
        )
        self.assertAlmostEqual(float(with_masking[0, 0]), 1.0, places=6)

    def test_shape_mismatch_raises(self):
        basis = vp.ortho_view_basis("front")
        center = np.zeros(3)
        image_size = 4
        depth_map = np.full((image_size, image_size), 5.0)
        wrong_shape_valid = np.ones((3, 3), dtype=bool)
        positions = np.zeros((1, 1, 3))
        normals = np.zeros((1, 1, 3))
        normals[0, 0] = basis.direction
        subject_mask = np.ones((image_size, image_size), dtype=bool)

        with self.assertRaises(ValueError):
            vp.view_texel_weights(
                positions, normals, basis, center, 2.0, image_size,
                depth_map, depth_bias=0.01, subject_mask=subject_mask, frame=(1.0, 0.0, 0.0),
                camera_distance=0.0, depth_map_valid=wrong_shape_valid,
            )


class SampleBilinearTests(unittest.TestCase):
    def test_exact_pixel_centers_are_returned_unchanged(self):
        image = np.zeros((2, 2, 3))
        image[0, 0] = (1.0, 0.0, 0.0)
        image[0, 1] = (0.0, 1.0, 0.0)
        image[1, 0] = (0.0, 0.0, 1.0)
        image[1, 1] = (1.0, 1.0, 1.0)
        result = vp.sample_bilinear(image, np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        np.testing.assert_allclose(result[0], (1.0, 0.0, 0.0))
        np.testing.assert_allclose(result[1], (1.0, 1.0, 1.0))

    def test_midpoint_averages_neighbors(self):
        image = np.zeros((2, 2, 3))
        image[0, 0] = (0.0, 0.0, 0.0)
        image[0, 1] = (1.0, 1.0, 1.0)
        image[1, 0] = (0.0, 0.0, 0.0)
        image[1, 1] = (1.0, 1.0, 1.0)
        result = vp.sample_bilinear(image, np.array([0.5]), np.array([0.0]))
        np.testing.assert_allclose(result[0], (0.5, 0.5, 0.5))

    def test_out_of_bounds_coordinates_are_clamped(self):
        image = np.zeros((2, 2, 3))
        image[0, 0] = (0.25, 0.5, 0.75)
        result = vp.sample_bilinear(image, np.array([-5.0]), np.array([-5.0]))
        np.testing.assert_allclose(result[0], (0.25, 0.5, 0.75))


class BlendViewsTests(unittest.TestCase):
    def test_disjoint_weights_produce_their_own_colors(self):
        red = np.zeros((2, 2, 3))
        red[..., 0] = 1.0
        blue = np.zeros((2, 2, 3))
        blue[..., 2] = 1.0
        weights_a = np.array([[1.0, 0.0], [1.0, 0.0]])
        weights_b = np.array([[0.0, 1.0], [0.0, 1.0]])
        fallback = np.full((2, 2, 3), 0.5)

        result, projected_mask = vp.blend_views([red, blue], [weights_a, weights_b], fallback)
        self.assertTrue(projected_mask.all())
        np.testing.assert_allclose(result[:, 0], red[:, 0])
        np.testing.assert_allclose(result[:, 1], blue[:, 1])

    def test_zero_total_weight_falls_back(self):
        color = np.ones((2, 2, 3))
        weights = np.zeros((2, 2))
        fallback = np.full((2, 2, 3), 0.5)
        result, projected_mask = vp.blend_views([color], [weights], fallback)
        self.assertFalse(projected_mask.any())
        np.testing.assert_allclose(result, fallback)

    def test_weighted_average_of_overlapping_views(self):
        color_a = np.full((1, 1, 3), 0.0)
        color_b = np.full((1, 1, 3), 1.0)
        weights_a = np.array([[1.0]])
        weights_b = np.array([[1.0]])
        fallback = np.zeros((1, 1, 3))
        result, projected_mask = vp.blend_views([color_a, color_b], [weights_a, weights_b], fallback)
        self.assertTrue(projected_mask[0, 0])
        np.testing.assert_allclose(result[0, 0], (0.5, 0.5, 0.5))


class ColorDeltaOklabTests(unittest.TestCase):
    def test_identical_colors_have_zero_delta(self):
        rgb = np.random.default_rng(0).uniform(0.0, 1.0, size=(4, 4, 3))
        mask = np.ones((4, 4), dtype=bool)
        delta = vp.color_delta_oklab(rgb, rgb, mask)
        self.assertAlmostEqual(delta, 0.0, places=9)

    def test_no_masked_texels_returns_none(self):
        rgb = np.zeros((2, 2, 3))
        mask = np.zeros((2, 2), dtype=bool)
        self.assertIsNone(vp.color_delta_oklab(rgb, rgb, mask))

    def test_different_colors_have_positive_delta(self):
        a = np.zeros((1, 1, 3))
        b = np.ones((1, 1, 3))
        mask = np.ones((1, 1), dtype=bool)
        delta = vp.color_delta_oklab(a, b, mask)
        self.assertGreater(delta, 0.0)


class ProjectionReportTests(unittest.TestCase):
    def test_keys_and_fractions(self):
        covered = np.ones((4, 4), dtype=bool)
        projected_mask = np.zeros((4, 4), dtype=bool)
        projected_mask[:2, :] = True
        per_role_coverage = {"front": 0.5, "back": 0.0, "left": 0.1, "right": 0.3}
        occlusion_rejected = {"front": 0.1, "back": 0.0, "left": 0.0, "right": 0.05}
        frames = {role: (1.0, 0.0, 0.0) for role in per_role_coverage}

        report = vp.projection_report(
            per_role_coverage, projected_mask, covered, occlusion_rejected, 0.02, 4, frames,
        )
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["views_used"], ["front", "left", "right"])
        self.assertAlmostEqual(report["projected_texel_fraction"], 0.5)
        self.assertAlmostEqual(report["fallback_texel_fraction"], 0.5)
        self.assertEqual(report["atlas_size"], 4)
        self.assertIn("back", report["frame_alignment"])
        self.assertEqual(report["frame_alignment"]["front"], {"scale": 1.0, "dx": 0.0, "dy": 0.0})
        self.assertIsInstance(report["caveats"], list)
        self.assertTrue(any("back" in caveat for caveat in report["caveats"]))

    def test_no_covered_texels_reports_zero_fractions_and_a_caveat(self):
        covered = np.zeros((2, 2), dtype=bool)
        projected_mask = np.zeros((2, 2), dtype=bool)
        report = vp.projection_report({}, projected_mask, covered, {}, None, 2, {})
        self.assertEqual(report["projected_texel_fraction"], 0.0)
        self.assertEqual(report["fallback_texel_fraction"], 0.0)
        self.assertIsNone(report["mean_color_delta_vs_bake"])
        self.assertTrue(report["caveats"])


class EndToEndSyntheticCubeTests(unittest.TestCase):
    """A unit cube whose six faces have distinct UV islands, projected from
    four synthetic solid-colour views with matching depth maps."""

    IMAGE_SIZE = 64
    ORTHO_SCALE = 1.16
    DEPTH_BIAS = 0.05

    FACE_COLORS = {
        "front": (1.0, 0.0, 0.0),  # -Y face
        "back": (0.0, 1.0, 0.0),  # +Y face
        "left": (0.0, 0.0, 1.0),  # +X face (camera direction is +X)
        "right": (1.0, 1.0, 0.0),  # -X face (camera direction is -X)
    }
    # Each face gets its own UV island, packed side by side across the U axis.
    FACE_ORDER = ["front", "back", "left", "right", "top", "bottom"]

    def _cube_face(self, normal_axis: int, sign: float):
        """Four corners (world) of a unit-cube face, normal along `normal_axis`."""
        other_axes = [axis for axis in range(3) if axis != normal_axis]
        corners = []
        for a in (-0.5, 0.5):
            for b in (-0.5, 0.5):
                point = [0.0, 0.0, 0.0]
                point[normal_axis] = sign * 0.5
                point[other_axes[0]] = a
                point[other_axes[1]] = b
                corners.append(point)
        # Order as a quad: (a=-.5,b=-.5) (a=-.5,b=.5) (a=.5,b=.5) (a=.5,b=-.5)
        return [corners[0], corners[1], corners[3], corners[2]]

    def _build_atlas_triangles(self):
        # Face -> (normal axis, sign), matching ROLE_AZIMUTH_DEGREES cameras:
        # front camera direction=(0,-1,0) sees the -Y face; back sees +Y;
        # left camera direction=(1,0,0) sees the +X face; right sees -X.
        face_geometry = {
            "front": (1, -1.0),
            "back": (1, 1.0),
            "left": (0, 1.0),
            "right": (0, -1.0),
            "top": (2, 1.0),
            "bottom": (2, -1.0),
        }
        uv_triangles = []
        world_triangles = []
        normal_triangles = []
        n_faces = len(self.FACE_ORDER)
        for index, face in enumerate(self.FACE_ORDER):
            axis, sign = face_geometry[face]
            quad = self._cube_face(axis, sign)
            normal = [0.0, 0.0, 0.0]
            normal[axis] = sign
            u0, u1 = index / n_faces, (index + 1) / n_faces
            uv_quad = [(u0, 0.0), (u0, 1.0), (u1, 1.0), (u1, 0.0)]
            for tri_indices in ((0, 1, 2), (0, 2, 3)):
                world_triangles.append([quad[i] for i in tri_indices])
                uv_triangles.append([uv_quad[i] for i in tri_indices])
                normal_triangles.append([normal, normal, normal])
        return (
            np.array(uv_triangles, dtype=np.float64),
            np.array(world_triangles, dtype=np.float64),
            np.array(normal_triangles, dtype=np.float64),
        )

    def test_blended_atlas_recovers_each_views_color_on_its_own_face(self):
        atlas_size = 256
        uv_triangles, world_triangles, normal_triangles = self._build_atlas_triangles()
        positions, normals, covered = vp.rasterize_atlas(uv_triangles, world_triangles, normal_triangles, atlas_size)
        self.assertTrue(covered.any())

        center = np.zeros(3)
        fallback_rgb = np.full((atlas_size, atlas_size, 3), 0.5)
        view_colors = []
        view_weights = []
        for role, color in self.FACE_COLORS.items():
            basis = vp.ortho_view_basis(role)
            image = np.tile(color, (self.IMAGE_SIZE, self.IMAGE_SIZE, 1)).astype(np.float64)
            subject_mask = np.ones((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=bool)
            # Every facing cube face sits at depth -0.5 relative to the
            # world origin: `depth = dot(point - center, forward)`, and each
            # camera's near face is half the cube's side BEHIND the origin
            # along its own forward axis.
            depth_map = np.full((self.IMAGE_SIZE, self.IMAGE_SIZE), -0.5)
            frame = (1.0, 0.0, 0.0)
            weights, sample_px, sample_py = vp.view_texel_weights(
                positions, normals, basis, center, self.ORTHO_SCALE, self.IMAGE_SIZE,
                depth_map, self.DEPTH_BIAS, subject_mask, frame, camera_distance=0.0,
            )
            colors = vp.sample_bilinear(image, sample_px, sample_py)
            view_colors.append(colors)
            view_weights.append(weights)

        blended, projected_mask = vp.blend_views(view_colors, view_weights, fallback_rgb)

        face_index = {face: i for i, face in enumerate(self.FACE_ORDER)}
        n_faces = len(self.FACE_ORDER)

        def face_center_pixel(face: str) -> tuple[int, int]:
            index = face_index[face]
            u = (index + 0.5) / n_faces
            v = 0.5
            col = int(u * atlas_size)
            row = int((1.0 - v) * atlas_size)
            return row, col

        for role, color in self.FACE_COLORS.items():
            row, col = face_center_pixel(role)
            self.assertTrue(projected_mask[row, col], f"{role} face was not projected")
            np.testing.assert_allclose(blended[row, col], color, atol=0.05)

        # A face none of the four views can see (top/bottom) keeps the
        # existing bake's fallback colour.
        for role in ("top", "bottom"):
            row, col = face_center_pixel(role)
            np.testing.assert_allclose(blended[row, col], (0.5, 0.5, 0.5), atol=1e-6)


class MinFacingTests(unittest.TestCase):
    """Pins bead s2-view-projection: a texel's weight is zeroed below
    `min_facing` even when its facing is positive (a foreshortened, grazing
    sample -- the "vertical smear streaks" finding), while the default 0.0
    reproduces every weight the old, cutoff-free formula produced.
    """

    def _weights_for_facings(self, facings, min_facing):
        basis = vp.ortho_view_basis("front")
        image_size = 8
        center = np.zeros(3)
        ortho_scale = 2.0
        n = len(facings)
        positions = np.zeros((1, n, 3))
        normals = np.zeros((1, n, 3))
        for index, facing in enumerate(facings):
            # basis.direction and basis.right are orthonormal, so this
            # normal has dot(normal, basis.direction) == facing exactly.
            sin_component = math.sqrt(max(0.0, 1.0 - facing ** 2))
            normals[0, index] = facing * basis.direction + sin_component * basis.right
        depth_map = np.zeros((image_size, image_size))
        subject_mask = np.ones((image_size, image_size), dtype=bool)
        weights, _, _ = vp.view_texel_weights(
            positions, normals, basis, center, ortho_scale, image_size,
            depth_map, depth_bias=float("inf"), subject_mask=subject_mask,
            frame=(1.0, 0.0, 0.0), camera_distance=0.0, min_facing=min_facing,
        )
        return weights[0]

    def test_weights_zero_below_cutoff_unchanged_above(self):
        facings = [-0.2, 0.0, 0.2, 0.34, 0.35, 0.5, 1.0]
        weights = self._weights_for_facings(facings, vp.DEFAULT_MIN_FACING)
        for facing, weight in zip(facings, weights):
            if facing < vp.DEFAULT_MIN_FACING:
                self.assertEqual(float(weight), 0.0, facing)
            else:
                self.assertAlmostEqual(float(weight), facing ** 2, places=9, msg=str(facing))

    def test_default_min_facing_reproduces_prior_weights_exactly(self):
        facings = [-0.5, -0.01, 0.0, 0.01, 0.3, 0.9, 1.0]
        weights = self._weights_for_facings(facings, 0.0)
        for facing, weight in zip(facings, weights):
            expected = max(facing, 0.0) ** 2
            self.assertAlmostEqual(float(weight), expected, places=9, msg=str(facing))


class MatchViewColourToReferenceTests(unittest.TestCase):
    def test_scaled_and_offset_view_maps_back_to_reference(self):
        rng = np.random.default_rng(0)
        reference_rgb = rng.uniform(0.2, 0.8, size=(32, 32, 3))
        view_rgb = 0.95 * reference_rgb + 0.1  # gain ~1.05, inside the 0.9-1.1 clamp
        weights = np.ones((32, 32))
        matched, info = vp.match_view_colour_to_reference(view_rgb, weights, reference_rgb, weights)
        np.testing.assert_allclose(matched, reference_rgb, atol=1e-6)
        self.assertTrue(info["applied"])
        self.assertEqual(info["overlap"], 32 * 32)

    def test_insufficient_overlap_is_identity(self):
        reference_rgb = np.full((32, 32, 3), 0.5)
        view_rgb = np.full((32, 32, 3), 0.9)
        view_weights = np.zeros((32, 32))
        view_weights[:5, :5] = 1.0  # 25 overlap texels, below the default min_overlap=500
        reference_weights = np.ones((32, 32))
        matched, info = vp.match_view_colour_to_reference(view_rgb, view_weights, reference_rgb, reference_weights)
        np.testing.assert_array_equal(matched, view_rgb)
        self.assertFalse(info["applied"])
        self.assertEqual(info["overlap"], 25)
        self.assertEqual(info["reason"], "insufficient_overlap")

    def test_gain_clamp_respected(self):
        rng = np.random.default_rng(1)
        reference_rgb = rng.uniform(0.0, 1.0, size=(32, 32, 3))
        mean = reference_rgb.mean(axis=(0, 1), keepdims=True)
        # 0.85x the reference's spread -> an unclamped gain would be ~1.18;
        # clamped to 1.1 the match still removes most of the residual, so it
        # applies. The clamp is deliberately tight: a view's drift is a colour
        # OFFSET, and a 0.5 gain flattened a whole view's detail on 2026-09-05.
        view_rgb = mean + (reference_rgb - mean) * 0.85
        weights = np.ones((32, 32))
        _, info = vp.match_view_colour_to_reference(view_rgb, weights, reference_rgb, weights)
        self.assertTrue(info["applied"])
        for gain in info["gain"]:
            self.assertLessEqual(gain, 1.1)
            self.assertGreaterEqual(gain, 0.9)

    def test_uncorrelated_overlap_is_identity(self):
        # Overlap texels that are NOT the same surface seen twice (the
        # conformance fixture's colour-region boundaries): a statistics
        # match cannot reduce the residual, so it must decline rather than
        # distort the whole view (gains 2.0 / 0.5 cost 18 points of
        # classification accuracy on 2026-09-05).
        rng = np.random.default_rng(3)
        reference_rgb = rng.uniform(0.0, 1.0, size=(32, 32, 3))
        view_rgb = rng.uniform(0.0, 1.0, size=(32, 32, 3))
        weights = np.ones((32, 32))
        matched, info = vp.match_view_colour_to_reference(view_rgb, weights, reference_rgb, weights)
        np.testing.assert_array_equal(matched, view_rgb)
        self.assertFalse(info["applied"])
        self.assertEqual(info["reason"], "no_residual_gain")
        self.assertGreater(info["residual_after"], 0.7 * info["residual_before"])

    def test_zero_variance_view_is_identity(self):
        reference_rgb = np.random.default_rng(2).uniform(0.0, 1.0, size=(32, 32, 3))
        view_rgb = np.full((32, 32, 3), 0.5)  # perfectly flat -> zero std
        weights = np.ones((32, 32))
        matched, info = vp.match_view_colour_to_reference(view_rgb, weights, reference_rgb, weights)
        np.testing.assert_array_equal(matched, view_rgb)
        self.assertFalse(info["applied"])
        self.assertEqual(info["reason"], "zero_variance")


class DilateProjectedTests(unittest.TestCase):
    def test_radius_grows_exact_chebyshev_ring_and_fills_uniform_colour(self):
        size = 64
        square = np.zeros((size, size), dtype=bool)
        square[27:37, 27:37] = True
        square_color = np.array([0.2, 0.4, 0.6])
        fallback_color = np.array([0.9, 0.1, 0.1])
        blended_rgb = np.full((size, size, 3), fallback_color)
        blended_rgb[square] = square_color
        fallback_rgb = np.full((size, size, 3), fallback_color)

        radius = 3
        rgb, filled_mask = vp.dilate_projected(blended_rgb, square, fallback_rgb, radius=radius)

        expanded = np.zeros((size, size), dtype=bool)
        expanded[27 - radius:37 + radius, 27 - radius:37 + radius] = True
        expected_filled = expanded & ~square

        np.testing.assert_array_equal(filled_mask, expected_filled)
        np.testing.assert_allclose(
            rgb[expected_filled], np.broadcast_to(square_color, (int(expected_filled.sum()), 3)), atol=1e-9,
        )
        beyond = ~expanded
        np.testing.assert_allclose(
            rgb[beyond], np.broadcast_to(fallback_color, (int(beyond.sum()), 3)), atol=1e-9,
        )

    def test_covered_restricts_growth(self):
        size = 20
        square = np.zeros((size, size), dtype=bool)
        square[8:12, 8:12] = True
        color = np.array([0.3, 0.3, 0.3])
        fallback_color = np.array([0.0, 0.0, 0.0])
        blended_rgb = np.full((size, size, 3), fallback_color)
        blended_rgb[square] = color
        fallback_rgb = np.full((size, size, 3), fallback_color)
        covered = square.copy()  # no room to grow beyond the square itself

        rgb, filled_mask = vp.dilate_projected(blended_rgb, square, fallback_rgb, radius=5, covered=covered)
        self.assertFalse(filled_mask.any())
        np.testing.assert_allclose(rgb, blended_rgb)

    def test_radius_zero_is_a_no_op(self):
        size = 10
        square = np.zeros((size, size), dtype=bool)
        square[3:6, 3:6] = True
        color = np.array([0.1, 0.2, 0.3])
        fallback_color = np.array([0.9, 0.8, 0.7])
        blended_rgb = np.full((size, size, 3), fallback_color)
        blended_rgb[square] = color
        fallback_rgb = np.full((size, size, 3), fallback_color)

        rgb, filled_mask = vp.dilate_projected(blended_rgb, square, fallback_rgb, radius=0)
        self.assertFalse(filled_mask.any())
        np.testing.assert_allclose(rgb, blended_rgb)


class RolePriorityWeightingTests(unittest.TestCase):
    """Role priority is applied by the caller (multiplying each role's
    weights before `blend_views`); pins that this produces exactly the
    proportional blend the priority ratio implies, using the real defaults.
    """

    def test_priority_weighted_blend_favors_higher_priority_view(self):
        color_a = np.full((1, 1, 3), 0.0)
        color_b = np.full((1, 1, 3), 1.0)
        facing_weight = np.array([[1.0]])
        priority_a = vp.DEFAULT_ROLE_PRIORITY["front"]
        priority_b = vp.DEFAULT_ROLE_PRIORITY["left"]
        self.assertEqual((priority_a, priority_b), (1.0, 0.5))
        weights_a = facing_weight * priority_a
        weights_b = facing_weight * priority_b
        fallback = np.zeros((1, 1, 3))

        result, projected_mask = vp.blend_views([color_a, color_b], [weights_a, weights_b], fallback)
        self.assertTrue(projected_mask[0, 0])
        # weight_a:weight_b is 2:1, so the blend sits 2/3 of the way from
        # color_b toward color_a: (0*1.0 + 1*0.5) / 1.5 == 1/3.
        np.testing.assert_allclose(result[0, 0], (1.0 / 3.0,) * 3, atol=1e-9)


def _tpose_mask(size: int = 100) -> np.ndarray:
    """A TOP-DOWN stick T-pose (row 0 = top of the image): head (rows 10-24,
    cols 45-55), torso (rows 25-89, cols 42-58) and arms (rows 26-31, cols
    5-95). The arms cross the top 20 % band (rows 10-25) from below; only the
    head enters it from above. Use `_tpose_mask_bottom_up` for the module's
    native (Blender `image.pixels`) orientation."""
    mask = np.zeros((size, size), dtype=bool)
    mask[10:25, 45:55] = True
    mask[25:90, 42:58] = True
    mask[26:32, 5:95] = True
    return mask


def _tpose_mask_bottom_up(size: int = 100) -> np.ndarray:
    """The same T-pose in the module's BOTTOM-UP row convention (row 0 = the
    bottom of the image, as `_load_reference_pixels` produces): boots at rows
    10-13, torso rows 10-74, arms rows 68-73, head rows 75-89 -- the head is
    at the HIGH row indices."""
    return _tpose_mask(size)[::-1].copy()


class HeadBboxTests(unittest.TestCase):
    # -- module-native bottom-up masks (the default, what the Blender worker passes) --

    def test_default_orientation_is_bottom_up_and_the_box_covers_the_head_rows(self):
        mask = _tpose_mask_bottom_up()
        box = vp.head_bbox(mask)
        # the head occupies rows 75-89 and the band (top 20 % of the 80-row
        # subject) reaches down to the neck row 74, so the box is rows 74-89
        # inclusive; the mirror of the top-down (45, 10, 55, 26) box
        self.assertEqual(box, (45, 74, 55, 90))
        self.assertEqual(box, vp.head_bbox(mask, rows_bottom_up=True))
        x0, y0, x1, y1 = box
        self.assertTrue(mask[75:90, 45:55].all())
        self.assertTrue(y0 <= 75 and y1 >= 90, "box must cover every head row")
        # and it must NOT be the boots (rows 10-25 of the bottom-up buffer)
        self.assertGreater(y0, 25)

    def test_bottom_up_exclude_head_removes_the_head_and_keeps_the_feet(self):
        mask = _tpose_mask_bottom_up()
        cleared, box = vp.exclude_head(mask)
        self.assertEqual(box, (45, 74, 55, 90))
        self.assertFalse(cleared[74:90, 45:55].any())
        # the feet (low row indices) must still project
        self.assertTrue(cleared[10:26, 42:58].all())
        outside = np.ones_like(mask)
        outside[74:90, 45:55] = False
        np.testing.assert_array_equal(cleared[outside], mask[outside])
        self.assertTrue(mask[74:90, 45:55].all())  # input not mutated

    def test_bottom_up_arms_lifted_into_the_band_do_not_widen_the_box(self):
        mask = _tpose_mask_bottom_up()
        mask[68:76, 5:95] = True  # arms lifted into the band's lower rows
        box = vp.head_bbox(mask)
        self.assertEqual(box[0], 45)
        self.assertEqual(box[2], 55)

    def test_bottom_up_result_mirrors_the_top_down_result(self):
        top_down = _tpose_mask()
        height = top_down.shape[0]
        x0, y0, x1, y1 = vp.head_bbox(top_down, rows_bottom_up=False)
        self.assertEqual(vp.head_bbox(top_down[::-1]), (x0, height - y1, x1, height - y0))

    # -- explicit top-down masks (row 0 = top of the image) --

    def test_top_down_head_box_covers_the_head_not_the_arms(self):
        box = vp.head_bbox(_tpose_mask(), rows_bottom_up=False)
        # the band is the top 20 % of the 80-row subject (rows 10-25 inclusive),
        # so the box runs to the neck row, like the gate's head box
        self.assertEqual(box, (45, 10, 55, 26))

    def test_top_down_arms_that_reach_into_the_band_bottom_do_not_widen_the_box(self):
        mask = _tpose_mask()
        mask[24:32, 5:95] = True  # arms lifted into the band's lower rows
        box = vp.head_bbox(mask, rows_bottom_up=False)
        self.assertEqual(box[0], 45)
        self.assertEqual(box[2], 55)

    def test_top_down_mask_with_default_orientation_would_pick_the_feet(self):
        # documents the bug the orientation flag exists for: a top-down mask
        # fed through the bottom-up default lands on the boots, not the head
        box = vp.head_bbox(_tpose_mask())
        self.assertLess(box[1], 90)
        self.assertGreaterEqual(box[1], 74)
        self.assertNotEqual(box, (45, 10, 55, 26))

    def test_empty_mask_returns_none(self):
        self.assertIsNone(vp.head_bbox(np.zeros((8, 8), dtype=bool)))
        self.assertIsNone(vp.head_bbox(np.zeros((8, 8), dtype=bool), rows_bottom_up=False))

    def test_top_down_exclude_head_clears_only_the_head_box(self):
        mask = _tpose_mask()
        cleared, box = vp.exclude_head(mask, rows_bottom_up=False)
        # the band is the top 20 % of the 80-row subject (rows 10-25 inclusive),
        # so the box runs to the neck row, like the gate's head box
        self.assertEqual(box, (45, 10, 55, 26))
        self.assertFalse(cleared[10:26, 45:55].any())
        # everything outside the box is untouched, and the input is not mutated
        outside = np.ones_like(mask)
        outside[10:26, 45:55] = False
        np.testing.assert_array_equal(cleared[outside], mask[outside])
        self.assertTrue(mask[10:26, 45:55].all())

    def test_exclude_head_on_empty_mask_is_a_noop(self):
        cleared, box = vp.exclude_head(np.zeros((8, 8), dtype=bool))
        self.assertIsNone(box)
        self.assertFalse(cleared.any())


class TiltedViewBasisTests(unittest.TestCase):
    """The perspective path's basis must be the SAME frame `SourceCamera`
    describes and Blender's `to_track_quat("-Z", "Y")` builds -- a mirrored
    or wrongly-pitched basis projects a photo onto the wrong side of the mesh
    and every numpy-only assertion below is what catches it before the box."""

    def test_zero_pitch_reduces_to_the_azimuth_basis(self):
        for yaw in (0.0, 45.0, 90.0, -135.0):
            tilted = vp.tilted_view_basis("photo", yaw, 0.0)
            plain = vp.azimuth_view_basis("photo", yaw)
            np.testing.assert_allclose(tilted.direction, plain.direction, atol=1e-12)
            np.testing.assert_allclose(tilted.right, plain.right, atol=1e-12)
            np.testing.assert_allclose(tilted.up, plain.up, atol=1e-12)
            np.testing.assert_allclose(tilted.forward, plain.forward, atol=1e-12)

    def test_direction_matches_source_camera(self):
        for yaw, pitch in ((0.0, 0.0), (35.0, 12.0), (-70.0, -8.0), (150.0, 25.0)):
            basis = vp.tilted_view_basis("photo", yaw, pitch)
            camera = SourceCamera(
                kind="perspective", width=8, height=8, center=(0.0, 0.0, 0.0),
                distance=3.0, yaw_degrees=yaw, pitch_degrees=pitch, fov_x_degrees=40.0,
            )
            np.testing.assert_allclose(basis.direction, camera.direction(), atol=1e-12)

    def test_orthonormal_right_handed_and_right_stays_horizontal(self):
        basis = vp.tilted_view_basis("photo", 35.0, 20.0)
        for vector in (basis.right, basis.up, basis.forward):
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=12)
        self.assertAlmostEqual(float(basis.right[2]), 0.0, places=12)
        np.testing.assert_allclose(basis.up, np.cross(basis.right, basis.forward), atol=1e-12)
        # A camera pitched DOWN at the subject still has world +Z above the
        # horizon in its image (up_z > 0), never below it.
        self.assertGreater(float(basis.up[2]), 0.0)

    def test_pitch_up_looks_down_from_above(self):
        basis = vp.tilted_view_basis("photo", 0.0, 30.0)
        self.assertGreater(float(basis.direction[2]), 0.0)
        self.assertLess(float(basis.forward[2]), 0.0)

    def test_straight_down_raises(self):
        with self.assertRaises(ValueError):
            vp.tilted_view_basis("photo", 0.0, 90.0)


class ProjectPointsPerspectiveTests(unittest.TestCase):
    WIDTH = 200
    HEIGHT = 200
    FOV_X = 40.0
    DISTANCE = 5.0

    def _basis(self, yaw=0.0, pitch=0.0):
        return vp.tilted_view_basis("photo", yaw, pitch)

    def _focal_px(self, width=None):
        width = self.WIDTH if width is None else width
        return (width / 2.0) / math.tan(math.radians(self.FOV_X) / 2.0)

    def _project(self, points, basis=None, width=None, height=None, center=None):
        return vp.project_points_perspective(
            np.asarray(points, dtype=np.float64),
            basis if basis is not None else self._basis(),
            np.zeros(3) if center is None else center,
            self.DISTANCE,
            self.FOV_X,
            self.WIDTH if width is None else width,
            self.HEIGHT if height is None else height,
        )

    def test_point_on_the_view_axis_projects_to_the_image_centre(self):
        # The subject centre and a point in front of it along the same axis.
        px, py, _ = self._project([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        np.testing.assert_allclose(px, [self.WIDTH / 2.0, self.WIDTH / 2.0], atol=1e-9)
        np.testing.assert_allclose(py, [self.HEIGHT / 2.0, self.HEIGHT / 2.0], atol=1e-9)

    def test_lateral_offset_projects_at_focal_times_offset_over_distance(self):
        # Yaw-0 camera sits at -Y looking +Y: world +X is its right, world +Z
        # its up, and both scale by f / (distance along the view axis).
        offset = 0.4
        depth_along_axis = self.DISTANCE - 1.0
        point = [offset, -1.0, offset]
        px, py, _ = self._project([point])
        focal = self._focal_px()
        self.assertAlmostEqual(
            float(px[0]), self.WIDTH / 2.0 + offset * focal / depth_along_axis, places=9,
        )
        self.assertAlmostEqual(
            float(py[0]), self.HEIGHT / 2.0 + offset * focal / depth_along_axis, places=9,
        )

    def test_depth_is_distance_along_the_view_axis_from_the_centre(self):
        # Same reference frame as the ortho path: Blender's Z pass for this
        # camera is `depth + camera_distance` (module docstring).
        points = np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.5, 0.0], [2.0, -1.0, 0.0]])
        _, _, depth = self._project(points)
        np.testing.assert_allclose(depth, [0.0, -1.0, 1.5, -1.0], atol=1e-9)
        np.testing.assert_allclose(depth + self.DISTANCE, [5.0, 4.0, 6.5, 4.0], atol=1e-9)

    def test_a_point_on_the_minus_y_side_is_in_front_of_the_yaw_zero_camera(self):
        # SourceCamera's convention: the canonical front faces world -Y and
        # yaw 0 photographs it, so the -Y half of the subject is the NEAR
        # half -- camera-relative depth below `camera_distance`, +Y above it.
        _, _, depth = self._project([[0.0, -0.5, 0.0], [0.0, 0.5, 0.0]])
        self.assertLess(float(depth[0] + self.DISTANCE), self.DISTANCE)
        self.assertGreater(float(depth[1] + self.DISTANCE), self.DISTANCE)

    def test_horizontal_handedness_matches_the_orthographic_path(self):
        # Same assertion the ortho tests make: world +X lands at a larger
        # column for the yaw-0 camera, world +Y for the yaw-90 (left) one.
        # The reversed cross product mirrors the photo onto the mesh.
        for yaw, plus_axis in ((0.0, [0.3, 0.0, 0.0]), (90.0, [0.0, 0.3, 0.0]),
                               (180.0, [-0.3, 0.0, 0.0]), (-90.0, [0.0, -0.3, 0.0])):
            px, _, _ = self._project([[0.0, 0.0, 0.0], plus_axis], basis=self._basis(yaw))
            self.assertGreater(float(px[1]), float(px[0]), f"yaw {yaw}")

    def test_world_up_maps_to_an_increasing_row(self):
        # Bottom-up rows, exactly as `project_points` documents.
        _, py, _ = self._project([[0.0, 0.0, 0.0], [0.0, 0.0, 0.3]])
        self.assertGreater(float(py[1]), float(py[0]))

    def test_non_square_image_shares_one_focal_length(self):
        # Square pixels: the horizontal FOV fixes the focal length and the
        # vertical FOV follows from the height, so a vertical offset uses the
        # SAME f -- getting this wrong stretches a portrait photo's overlay.
        width, height = 300, 100
        offset = 0.4
        px, py, _ = self._project([[offset, -1.0, offset]], width=width, height=height)
        focal = self._focal_px(width)
        depth_along_axis = self.DISTANCE - 1.0
        self.assertAlmostEqual(float(px[0]), width / 2.0 + offset * focal / depth_along_axis, places=9)
        self.assertAlmostEqual(float(py[0]), height / 2.0 + offset * focal / depth_along_axis, places=9)

    def test_points_at_or_behind_the_camera_are_reported_out_of_frame(self):
        # Finite, far out of frame: an inf would cast to a garbage int64 in
        # the bounds tests `_view_texel_masks` runs on these coordinates.
        px, py, depth = self._project([[0.0, -self.DISTANCE, 0.0], [0.0, -2 * self.DISTANCE, 0.0]])
        np.testing.assert_array_equal(px, [vp.BEHIND_CAMERA_PIXEL] * 2)
        np.testing.assert_array_equal(py, [vp.BEHIND_CAMERA_PIXEL] * 2)
        self.assertTrue(np.all(np.isfinite(depth)))
        self.assertEqual(int(np.round(px[0])), int(vp.BEHIND_CAMERA_PIXEL))

    def test_pitched_camera_sees_the_top_of_the_subject_above_centre(self):
        # A camera pitched 30 deg above the subject still puts a point on the
        # subject's own +Z axis ABOVE the image centre.
        _, py, _ = self._project([[0.0, 0.0, 0.5]], basis=self._basis(pitch=30.0))
        self.assertGreater(float(py[0]), self.HEIGHT / 2.0)


class PerspectiveCameraTests(unittest.TestCase):
    def _camera(self, width=200, height=200, pitch=0.0):
        return vp.PerspectiveCamera(
            basis=vp.tilted_view_basis("photo", 0.0, pitch),
            center=np.zeros(3),
            fov_x_degrees=40.0,
            camera_distance=5.0,
            width=width,
            height=height,
        )

    def test_location_is_center_plus_direction_times_distance(self):
        camera = self._camera()
        np.testing.assert_allclose(camera.location(), [0.0, -5.0, 0.0], atol=1e-9)

    def test_project_matches_the_free_function(self):
        camera = self._camera()
        points = np.array([[0.2, -0.4, 0.1], [-0.3, 0.5, -0.2]])
        expected = vp.project_points_perspective(
            points, camera.basis, camera.center, camera.camera_distance,
            camera.fov_x_degrees, camera.width, camera.height,
        )
        for actual_axis, expected_axis in zip(camera.project(points), expected):
            np.testing.assert_allclose(actual_axis, expected_axis, atol=1e-12)

    def test_facing_is_measured_per_point_not_along_one_axis(self):
        # A texel off to the side of the frame is seen along a ray tilted
        # away from the camera axis, so a surface whose normal points along
        # that ray faces the camera FULLY (1.0) even though its dot with
        # `basis.direction` is smaller -- the ortho approximation.
        camera = self._camera()
        point = np.array([[1.0, 0.0, 0.0]])
        toward_camera = camera.location() - point[0]
        toward_camera = toward_camera / np.linalg.norm(toward_camera)
        facing = camera.facing(toward_camera[None, :], point)
        self.assertAlmostEqual(float(facing[0]), 1.0, places=9)
        along_axis = float(toward_camera @ camera.basis.direction)
        self.assertLess(along_axis, 0.99)

    def test_world_units_per_pixel_covers_the_frame_width_at_the_subject(self):
        camera = self._camera()
        frame_width = 2.0 * camera.camera_distance * math.tan(math.radians(camera.fov_x_degrees) / 2.0)
        self.assertAlmostEqual(camera.world_units_per_pixel() * camera.width, frame_width, places=9)

    def test_world_units_per_pixel_grows_linearly_with_depth_beyond_the_subject(self):
        # the far plane of a subject of radius r sees pixels (d + r) / d wider than the centre plane
        camera = self._camera()
        near, far = camera.world_units_per_pixel(), camera.world_units_per_pixel(depth_offset=0.5)
        self.assertAlmostEqual(far / near, (camera.camera_distance + 0.5) / camera.camera_distance, places=9)
        ortho = vp.OrthographicCamera(
            basis=vp.ortho_view_basis("front"), center=np.zeros(3), ortho_scale=2.0,
            camera_distance=4.0, width=64, height=64,
        )
        self.assertEqual(ortho.world_units_per_pixel(depth_offset=0.5), ortho.world_units_per_pixel())

    def test_orthographic_camera_facing_is_the_constant_direction(self):
        basis = vp.ortho_view_basis("front")
        camera = vp.OrthographicCamera(
            basis=basis, center=np.zeros(3), ortho_scale=2.0,
            camera_distance=4.0, width=64, height=64,
        )
        normals = np.array([basis.direction, -basis.direction])
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        np.testing.assert_allclose(camera.facing(normals, points), [1.0, -1.0], atol=1e-12)
        self.assertAlmostEqual(camera.world_units_per_pixel(), 2.0 / 64.0, places=12)


class PerspectiveTexelWeightsTests(unittest.TestCase):
    """`view_texel_weights_for_camera` end to end with a perspective camera:
    a flat plate at the subject centre, its own analytic depth pass, and a
    subject mask that covers the frame."""

    WIDTH = 128
    HEIGHT = 96
    FOV_X = 45.0
    DISTANCE = 4.0

    def _camera(self):
        return vp.PerspectiveCamera(
            basis=vp.tilted_view_basis("photo", 0.0, 0.0),
            center=np.zeros(3),
            fov_x_degrees=self.FOV_X,
            camera_distance=self.DISTANCE,
            width=self.WIDTH,
            height=self.HEIGHT,
        )

    def _plate(self):
        """A 3x3 patch of atlas texels on the z=0 plane facing the camera."""
        xs = np.linspace(-0.3, 0.3, 3)
        zs = np.linspace(-0.3, 0.3, 3)
        positions = np.zeros((3, 3, 3))
        positions[:, :, 0] = xs[None, :]
        positions[:, :, 2] = zs[:, None]
        normals = np.zeros((3, 3, 3))
        normals[:, :, 1] = -1.0
        return positions, normals

    def test_visible_plate_texels_carry_weight_and_sample_inside_the_frame(self):
        camera = self._camera()
        positions, normals = self._plate()
        # The plate lies in the plane through `center`, so Blender's Z pass
        # for every pixel that hits it is exactly `camera_distance`.
        depth_map = np.full((self.HEIGHT, self.WIDTH), self.DISTANCE)
        subject_mask = np.ones((self.HEIGHT, self.WIDTH), dtype=bool)
        weights, sample_px, sample_py = vp.view_texel_weights_for_camera(
            positions, normals, camera, depth_map, 0.01, subject_mask, (1.0, 0.0, 0.0),
        )
        self.assertTrue(np.all(weights > 0.0))
        self.assertTrue(np.all((sample_px >= 0) & (sample_px < self.WIDTH)))
        self.assertTrue(np.all((sample_py >= 0) & (sample_py < self.HEIGHT)))
        # The centre texel is dead centre of a non-square frame.
        self.assertAlmostEqual(float(sample_px[1, 1]), self.WIDTH / 2.0, places=9)
        self.assertAlmostEqual(float(sample_py[1, 1]), self.HEIGHT / 2.0, places=9)

    def test_a_texel_behind_the_nearest_surface_is_occluded(self):
        camera = self._camera()
        positions, normals = self._plate()
        # The depth pass says the nearest surface is a full unit nearer than
        # the plate: every plate texel is behind it.
        depth_map = np.full((self.HEIGHT, self.WIDTH), self.DISTANCE - 1.0)
        subject_mask = np.ones((self.HEIGHT, self.WIDTH), dtype=bool)
        weights, _, _ = vp.view_texel_weights_for_camera(
            positions, normals, camera, depth_map, 0.01, subject_mask, (1.0, 0.0, 0.0),
        )
        self.assertTrue(np.all(weights == 0.0))

    def test_depth_map_shaped_for_a_square_image_is_rejected(self):
        camera = self._camera()
        positions, normals = self._plate()
        with self.assertRaisesRegex(ValueError, r"\(96, 128\).*\(128, 128\)"):
            vp.view_texel_weights_for_camera(
                positions, normals, camera, np.full((128, 128), self.DISTANCE), 0.01,
                np.ones((self.HEIGHT, self.WIDTH), dtype=bool), (1.0, 0.0, 0.0),
            )

    def test_diagnostics_and_debug_masks_accept_a_perspective_camera(self):
        camera = self._camera()
        positions, normals = self._plate()
        depth_map = np.full((self.HEIGHT, self.WIDTH), self.DISTANCE)
        subject_mask = np.ones((self.HEIGHT, self.WIDTH), dtype=bool)
        covered = np.ones((3, 3), dtype=bool)
        diagnostics = vp.view_texel_diagnostics_for_camera(
            positions, normals, camera, depth_map, 0.01, subject_mask, (1.0, 0.0, 0.0), covered,
        )
        self.assertEqual(diagnostics["n_covered"], 9)
        self.assertEqual(diagnostics["n_weight_positive"], 9)
        masks = vp.view_texel_debug_masks_for_camera(
            positions, normals, camera, depth_map, 0.01, subject_mask, (1.0, 0.0, 0.0),
        )
        self.assertTrue(masks["in_frame"].all())
        self.assertTrue(masks["depth_visible"].all())


if __name__ == "__main__":
    unittest.main()
