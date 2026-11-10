import hashlib
import json
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from backend.app.image_preparation import (
    ImagePreparationProfile,
    MESH_GUIDED_PLAIN_VIEW_SYNTHESIS_PROFILES,
    MESH_GUIDED_VIEW_SYNTHESIS_PROFILES,
    POSE_GUIDES,
    SOURCE_POSE_VIEW_SYNTHESIS_PROFILES,
    VIEW_SYNTHESIS_PROFILE_FAMILIES,
    VIEW_SYNTHESIS_PROFILES,
    build_flux2_klein_image_edit,
    build_head_crop_reference,
    build_view_synthesis_metadata,
    detect_subject_bbox,
    resolve_preparation_profile,
    resolve_view_synthesis_profile,
    run_comfyui_image_edit,
    t_pose_guide_asset,
)


def _synthetic_reference_png(
    *, size=(100, 100), background=(200, 200, 200), subject_box=(30, 10, 70, 90), subject=(20, 20, 20)
) -> bytes:
    """A plain-background image with a solid rectangular "subject" -- enough
    to exercise background/foreground bbox detection without needing a real
    render."""
    image = Image.new("RGB", size, background)
    for y in range(subject_box[1], subject_box[3]):
        for x in range(subject_box[0], subject_box[2]):
            image.putpixel((x, y), subject)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class ImagePreparationTests(unittest.TestCase):
    def test_sword_profile_expands_to_reproducible_pose_requirements(self):
        profile = resolve_preparation_profile("humanoid-a-pose-sword-v1")

        self.assertEqual(profile.id, "humanoid-a-pose-sword-v1")
        self.assertEqual(profile.version, 1)
        self.assertIn("full-body", profile.prompt)
        self.assertIn("A-pose", profile.prompt)
        self.assertIn("closed around a short neutral cylindrical grip", profile.prompt)
        self.assertIn("no sword blade", profile.negative_prompt)

    def test_unknown_preparation_profile_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Unsupported image preparation profile"):
            resolve_preparation_profile("magic")

    def test_v2_sword_profile_requires_exactly_one_right_hand_grip(self):
        profile = resolve_preparation_profile("humanoid-a-pose-sword-v2")

        self.assertEqual(profile.version, 2)
        self.assertIn("exactly one grip", profile.prompt)
        self.assertIn("right hand only", profile.prompt)
        self.assertIn("object in the left hand", profile.negative_prompt)

    def test_v3_rig_profile_defers_the_weapon_and_exposes_leg_joints(self):
        profile = resolve_preparation_profile("humanoid-a-pose-rig-v3")

        self.assertEqual(profile.version, 3)
        self.assertIn("both hands fully open and empty", profile.prompt)
        self.assertIn("hips, knees, and ankles", profile.prompt)
        self.assertIn("weapon", profile.negative_prompt)

    def test_t_pose_rig_profile_exposes_straight_limb_landmarks(self):
        profile = resolve_preparation_profile("humanoid-t-pose-rig-v1")

        self.assertEqual(profile.version, 3)
        self.assertIn("strict symmetrical T-pose", profile.prompt)
        self.assertIn("horizontal at shoulder height", profile.prompt)
        self.assertIn("elbows straight", profile.prompt)
        self.assertIn("both hands fully open and empty", profile.prompt)
        self.assertIn("hips, knees, and ankles", profile.prompt)
        self.assertIn("weapon", profile.negative_prompt)
        # The guide active in this checkout (assets/pose-guides/active.json,
        # or the AISMITH_T_POSE_GUIDE opt-in).
        self.assertEqual(profile.pose_guide_asset, t_pose_guide_asset())

    def test_the_active_t_pose_guide_exists_and_v2_is_an_explicit_opt_in(self):
        active = json.loads((POSE_GUIDES / "active.json").read_text(encoding="utf-8"))
        self.assertEqual(t_pose_guide_asset({}), f"assets/pose-guides/{active['humanoid-t-pose-rig-v1']}")
        self.assertTrue((POSE_GUIDES / active["humanoid-t-pose-rig-v1"]).is_file())
        self.assertEqual(
            t_pose_guide_asset({"AISMITH_T_POSE_GUIDE": "humanoid-t-pose-v2.png"}),
            "assets/pose-guides/humanoid-t-pose-v2.png",
        )
        for bad in ("missing.png", "../image_preparation.py", "sub/humanoid-t-pose-v2.png"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                t_pose_guide_asset({"AISMITH_T_POSE_GUIDE": bad})

    def test_the_cc0_t_pose_guide_has_a_provenance_record(self):
        provenance = (POSE_GUIDES / "PROVENANCE.md").read_text(encoding="utf-8")
        digest = hashlib.sha256((POSE_GUIDES / "humanoid-t-pose-v2.png").read_bytes()).hexdigest()
        self.assertIn(digest, provenance)

    def test_t_pose_v2_profile_hardens_outfit_and_identity_against_drift(self):
        # Operator escalation (2imyx-regression / bead ncxzd): the T-pose
        # candidate for an out-of-distribution alien source held pose and
        # baldness but drifted grey-green skin to pink human skin, purple
        # glowing eyes to blue, and adapted his tattered black coat into an
        # invented ornate navy-and-gold armor set. v2's clothing instruction
        # must name the reference garment explicitly instead of inviting
        # "fantasy clothing", and identity language must name skin tone and
        # eye color the way the view-synthesis v3/v4 profiles already do.
        profile = resolve_preparation_profile("humanoid-t-pose-rig-v1")

        self.assertIn("SAME garments worn in the first reference image", profile.prompt)
        self.assertIn("do not invent new clothing, armor, or accessories", profile.prompt.lower())
        self.assertIn("skin tone", profile.prompt)
        self.assertIn("eye color", profile.prompt)
        self.assertIn("bald or hairless", profile.prompt)
        self.assertIn("redesigned outfit", profile.negative_prompt)
        self.assertIn("ornate armor", profile.negative_prompt)
        self.assertIn("different skin tone", profile.negative_prompt)
        self.assertIn("human skin", profile.negative_prompt)
        self.assertIn("changed eye color", profile.negative_prompt)

    def test_t_pose_v3_profile_enumerates_garments_against_bodysuit_stripping(self):
        # Operator escalation (zix run): v2's "fitted closer to the body"
        # instruction plus a wall of anti-invention negatives gave the model
        # no positive anchor for what to render on that fitted silhouette --
        # two zix candidates in a row kept identity (face, circuitry, eyes)
        # but flattened the operator's tattered coat, hood, neck harness,
        # and belt into a blank charcoal bodysuit. v3 must keep the
        # loose/flowing-silhouette negatives (still a fitted, rig-ready
        # candidate) while explicitly enumerating the garment pieces and
        # surface details that must survive the fit-down, and must forbid
        # the blank-bodysuit failure mode by name.
        profile = resolve_preparation_profile("humanoid-t-pose-rig-v1")

        self.assertIn("ADAPTATION of the SAME", profile.prompt)
        self.assertIn("coat", profile.prompt.lower())
        self.assertIn("hood", profile.prompt.lower())
        self.assertIn("frayed", profile.prompt.lower())
        self.assertIn("collar, harness, or hardware", profile.prompt)
        self.assertIn("belt or waist strap", profile.prompt)
        self.assertIn("boots with their own texture and lacing", profile.prompt)
        self.assertIn("do not flatten the outfit into a single", profile.prompt.lower())
        # The fitted-silhouette requirement from v2 must still hold -- this
        # is a rebalance of what to keep, not a return to loose/flowing v1.
        self.assertIn("long dress, skirt, robe, cape", profile.negative_prompt)
        self.assertIn("wide fabric crossing hips", profile.negative_prompt)
        # New anti-stripping negatives.
        self.assertIn("plain bodysuit", profile.negative_prompt)
        self.assertIn("featureless bodysuit", profile.negative_prompt)
        self.assertIn("missing coat", profile.negative_prompt)
        self.assertIn("missing hood", profile.negative_prompt)
        self.assertIn("losing the belt", profile.negative_prompt)

    def test_t_pose_workflow_chains_identity_then_pose_references(self):
        profile = resolve_preparation_profile("humanoid-t-pose-rig-v1")

        workflow = build_flux2_klein_image_edit(
            input_name="mary-source.png",
            pose_input_name="humanoid-t-pose.png",
            output_prefix="aismith/mary-t-pose",
            profile=profile,
            seed=42,
        )

        load_images = {
            node_id: node
            for node_id, node in workflow.items()
            if node["class_type"] == "LoadImage"
        }
        self.assertEqual(
            {node["inputs"]["image"] for node in load_images.values()},
            {"mary-source.png", "humanoid-t-pose.png"},
        )
        self.assertEqual(workflow["92"]["inputs"]["conditioning"], ["76", 0])
        self.assertEqual(workflow["92"]["inputs"]["latent"], ["94", 0])
        self.assertEqual(workflow["63"]["inputs"]["positive"], ["92", 0])
        self.assertIn("identity and wardrobe reference", profile.prompt)
        self.assertIn("pose reference", profile.prompt)

    def test_t_pose_workflow_chains_pose_guide_onto_negative_symmetrically(self):
        # P1 image-adherence bug: the pose-guide latent was only ever chained
        # onto the POSITIVE conditioning (node 92), while the source-identity
        # latent is chained onto both positive (76) and negative (77). That
        # asymmetry means CFG's (positive - negative) differential is pulled
        # unopposed toward the pose guide's own generic mannequin appearance,
        # swamping the source character's identity. The pose latent must be
        # chained onto the negative conditioning too, exactly like the source
        # latent already is, so CFG stays pointed at the text instruction
        # (repose to T-pose) instead of at the guide image's own likeness.
        profile = resolve_preparation_profile("humanoid-t-pose-rig-v1")

        workflow = build_flux2_klein_image_edit(
            input_name="mary-source.png",
            pose_input_name="humanoid-t-pose.png",
            output_prefix="aismith/mary-t-pose",
            profile=profile,
            seed=42,
        )

        self.assertEqual(workflow["96"]["class_type"], "ReferenceLatent")
        self.assertEqual(workflow["96"]["inputs"]["conditioning"], ["77", 0])
        self.assertEqual(workflow["96"]["inputs"]["latent"], ["94", 0])
        self.assertEqual(workflow["63"]["inputs"]["negative"], ["96", 0])

    def test_non_pose_guide_workflow_leaves_negative_on_the_plain_text_node(self):
        # Profiles without a pose guide (no second reference image) must not
        # grow a spurious negative-side ReferenceLatent chain -- negative
        # stays wired directly to the negative CLIPTextEncode node (67).
        profile = resolve_preparation_profile("humanoid-a-pose-sword-v1")

        workflow = build_flux2_klein_image_edit(
            input_name="mary-source.png",
            output_prefix="aismith/mary-a-pose",
            profile=profile,
            seed=42,
        )

        self.assertNotIn("96", workflow)
        self.assertEqual(workflow["63"]["inputs"]["negative"], ["77", 0])

    def test_flux2_workflow_pins_models_seed_source_and_output(self):
        profile = resolve_preparation_profile("humanoid-a-pose-sword-v1")

        workflow = build_flux2_klein_image_edit(
            input_name="mary-source.png",
            output_prefix="aismith/mary-a-pose",
            profile=profile,
            seed=42,
        )

        by_type = {node["class_type"]: node for node in workflow.values()}

        self.assertEqual(
            by_type["UNETLoader"]["inputs"]["unet_name"],
            "flux-2-klein-base-4b-fp8.safetensors",
        )
        self.assertEqual(by_type["CLIPLoader"]["inputs"]["clip_name"], "qwen_3_4b.safetensors")
        self.assertEqual(by_type["VAELoader"]["inputs"]["vae_name"], "flux2-vae.safetensors")
        self.assertEqual(by_type["LoadImage"]["inputs"]["image"], "mary-source.png")
        self.assertEqual(by_type["RandomNoise"]["inputs"]["noise_seed"], 42)
        self.assertEqual(by_type["EmptyFlux2LatentImage"]["inputs"]["width"], profile.width)
        self.assertEqual(by_type["EmptyFlux2LatentImage"]["inputs"]["height"], profile.height)
        self.assertEqual(by_type["Flux2Scheduler"]["inputs"]["width"], profile.width)
        self.assertEqual(by_type["Flux2Scheduler"]["inputs"]["height"], profile.height)
        self.assertEqual(by_type["PreviewImage"]["inputs"]["images"], ["65", 0])
        encoded_prompts = [
            node["inputs"]["text"]
            for node in workflow.values()
            if node["class_type"] == "CLIPTextEncode"
        ]
        self.assertTrue(any(profile.prompt in prompt for prompt in encoded_prompts))
        self.assertTrue(any(profile.negative_prompt in prompt for prompt in encoded_prompts))

    def test_comfy_adapter_uploads_runs_and_downloads_the_generated_candidate(self):
        class Response:
            def __init__(self, payload=None, content=b""):
                self.payload = payload
                self.content = content

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        class Session:
            def __init__(self):
                self.posts = []

            def post(self, url, **kwargs):
                self.posts.append((url, kwargs))
                if url.endswith("/upload/image"):
                    return Response({"name": "mary-upload.png"})
                return Response({"prompt_id": "prompt-42", "node_errors": {}})

            def get(self, url, **kwargs):
                if "/history/" in url:
                    return Response({"prompt-42": {
                        "status": {"completed": True, "status_str": "success"},
                        "outputs": {"91": {"images": [{
                            "filename": "candidate.png", "subfolder": "aismith", "type": "output",
                        }]}},
                    }})
                return Response(content=b"generated-png")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mary.png"
            output = root / "candidate.png"
            source.write_bytes(b"source-png")
            session = Session()
            accepted = []

            metadata = run_comfyui_image_edit(
                source,
                output,
                profile=resolve_preparation_profile("humanoid-a-pose-sword-v1"),
                seed=42,
                base_url="http://comfy.test",
                session=session,
                on_accepted=accepted.append,
            )

            self.assertEqual(output.read_bytes(), b"generated-png")

        self.assertEqual(metadata["prompt_id"], "prompt-42")
        self.assertEqual(metadata["workflow_version"], "flux2-klein-image-edit-v1")
        self.assertEqual(metadata["model"], "flux-2-klein-base-4b-fp8.safetensors")
        self.assertEqual(accepted, ["prompt-42"])
        submitted = session.posts[1][1]["json"]["prompt"]
        self.assertEqual(submitted["73"]["inputs"]["noise_seed"], 42)
        uploaded_name = session.posts[0][1]["files"]["image"][0]
        self.assertRegex(uploaded_name, r"^aismith-preparation-[0-9a-f]{16}\.png$")

    def test_t_pose_adapter_uploads_pose_guide_and_records_its_provenance(self):
        class Response:
            def __init__(self, payload=None, content=b""):
                self.payload = payload
                self.content = content

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        class Session:
            def __init__(self):
                self.posts = []
                self.upload_count = 0

            def post(self, url, **kwargs):
                self.posts.append((url, kwargs))
                if url.endswith("/upload/image"):
                    self.upload_count += 1
                    return Response({"name": f"upload-{self.upload_count}.png"})
                return Response({"prompt_id": "prompt-42", "node_errors": {}})

            def get(self, url, **kwargs):
                if "/history/" in url:
                    return Response({"prompt-42": {
                        "status": {"completed": True, "status_str": "success"},
                        "outputs": {"91": {"images": [{
                            "filename": "candidate.png", "subfolder": "aismith", "type": "output",
                        }]}},
                    }})
                return Response(content=b"generated-png")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mary.png"
            pose_guide = root / resolve_preparation_profile("humanoid-t-pose-rig-v1").pose_guide_asset
            output = root / "candidate.png"
            source.write_bytes(b"source-png")
            pose_guide.parent.mkdir(parents=True)
            pose_guide.write_bytes(b"pose-guide-png")
            session = Session()

            metadata = run_comfyui_image_edit(
                source,
                output,
                profile=resolve_preparation_profile("humanoid-t-pose-rig-v1"),
                seed=42,
                base_url="http://comfy.test",
                session=session,
                pose_guide_root=root,
            )

        uploads = [post for post in session.posts if post[0].endswith("/upload/image")]
        self.assertEqual(len(uploads), 2)
        submitted = [post for post in session.posts if post[0].endswith("/prompt")][0][1]["json"]["prompt"]
        self.assertEqual(submitted["90"]["inputs"]["image"], "upload-1.png")
        self.assertEqual(submitted["93"]["inputs"]["image"], "upload-2.png")
        self.assertEqual(metadata["workflow_version"], "flux2-klein-multi-reference-image-edit-v2")
        self.assertRegex(metadata["pose_guide_sha256"], r"^[0-9a-f]{64}$")

    def test_comfy_validation_error_surfaces_node_details_before_http_error(self):
        class Response:
            ok = False
            content = b""

            def raise_for_status(self):
                raise RuntimeError("generic 400")

            def json(self):
                return {"node_errors": {"70": {"errors": ["model missing"]}}}

        class Session:
            def post(self, url, **_kwargs):
                if url.endswith("/upload/image"):
                    response = Response()
                    response.ok = True
                    response.json = lambda: {"name": "uploaded.png"}
                    response.raise_for_status = lambda: None
                    return response
                return Response()

        with TemporaryDirectory() as directory:
            source = Path(directory) / "mary.png"
            source.write_bytes(b"source")
            with self.assertRaisesRegex(RuntimeError, "model missing"):
                run_comfyui_image_edit(
                    source, Path(directory) / "candidate.png",
                    profile=resolve_preparation_profile("humanoid-a-pose-sword-v1"),
                    seed=1, base_url="http://comfy.test", session=Session(),
                )

    def test_cancelled_queued_comfy_prompt_is_removed_without_interrupting_another_client(self):
        class Response:
            ok = True
            content = b""

            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        class Session:
            def __init__(self):
                self.posts = []

            def post(self, url, **kwargs):
                self.posts.append((url, kwargs))
                if url.endswith("/upload/image"):
                    return Response({"name": "uploaded.png"})
                if url.endswith("/prompt"):
                    return Response({"prompt_id": "owned-prompt", "node_errors": {}})
                return Response({})

            def get(self, _url, **_kwargs):
                return Response({})

        with TemporaryDirectory() as directory:
            source = Path(directory) / "mary.png"
            source.write_bytes(b"source")
            session = Session()
            with self.assertRaisesRegex(InterruptedError, "cancelled"):
                run_comfyui_image_edit(
                    source, Path(directory) / "candidate.png",
                    profile=resolve_preparation_profile("humanoid-a-pose-sword-v1"),
                    seed=1, base_url="http://comfy.test", session=session,
                    cancelled=lambda: True,
                )

        self.assertTrue(any(url.endswith("/queue") for url, _ in session.posts))
        self.assertFalse(any(url.endswith("/interrupt") for url, _ in session.posts))


class ViewSynthesisTests(unittest.TestCase):
    def test_prompt_table_covers_back_left_right_with_distinct_ids(self):
        self.assertEqual(set(VIEW_SYNTHESIS_PROFILES), {"back", "left", "right"})
        ids = {profile.id for profile in VIEW_SYNTHESIS_PROFILES.values()}
        self.assertEqual(len(ids), 3, "each view synthesis profile must have a distinct id")

    def test_back_profile_demands_a_genuine_rear_view_not_a_mirrored_copy(self):
        profile = resolve_view_synthesis_profile("back")

        self.assertIn("directly behind", profile.prompt)
        self.assertIn("not a mirrored or flipped copy", profile.prompt)
        self.assertIn("same outfit", profile.prompt)
        self.assertIn("same lighting", profile.prompt)
        self.assertIn("mirrored copy of the reference image", profile.negative_prompt)
        self.assertIn("front view", profile.negative_prompt)

    def test_left_and_right_profiles_are_distinct_and_mutually_exclusive(self):
        left = resolve_view_synthesis_profile("left")
        right = resolve_view_synthesis_profile("right")

        self.assertIn("left side", left.prompt)
        self.assertIn("right-profile view", left.negative_prompt)
        self.assertIn("right side", right.prompt)
        self.assertIn("left-profile view", right.negative_prompt)
        self.assertNotEqual(left.prompt, right.prompt)

    def test_left_and_right_v5_profiles_anchor_direction_with_frame_geometry(self):
        # P1: v3's left/right prompts named the side ("left-profile view")
        # but gave no geometric anchor, so both views collapsed to the same
        # ~3/4 angle on an out-of-distribution source. v4 added a frame-edge
        # anchor but got the polarity backwards (working the camera-
        # placement geometry through by hand: a camera relocated to the
        # character's own left flank puts the nose on the frame's LEFT
        # edge, not the right -- independently confirmed against the
        # zix.webp source, whose own near-profile framing has the nose on
        # frame-left in exactly the shot that also shows the character's
        # left-side ear hardware) and shared a long identical preamble
        # across both prompts, and the live final-gate set showed both
        # LEFT and RIGHT collapsing to the SAME frame direction rather than
        # opposite ones. v5 corrects the polarity and front-loads the
        # direction-bearing camera-placement sentence as the FIRST
        # sentence of each prompt, so left/right diverge from the first
        # token instead of sharing an identical opening clause.
        left = resolve_view_synthesis_profile("left")
        right = resolve_view_synthesis_profile("right")

        self.assertIn("-v7", left.id)
        self.assertIn("-v7", right.id)
        self.assertIn("nose point toward the LEFT edge of the frame", left.prompt)
        self.assertIn("left ear and left temple face squarely toward the camera", left.prompt)
        self.assertIn("nose point toward the RIGHT edge of the frame", right.prompt)
        self.assertIn("right ear and right temple face squarely toward the camera", right.prompt)
        # The direction-bearing word must appear well before v4's ~140-
        # character identical shared preamble ended ("Use the reference
        # image only as the identity and wardrobe reference... Redraw this
        # exact same individual in a strict 90-degree [x]-profile view...").
        # v5's opening sentence is itself shared boilerplate ("Camera
        # placement is the defining requirement..."), so find where the two
        # prompts actually diverge and assert it lands well before v4's
        # preamble length, not after it.
        divergence_index = next(
            index for index, (a, b) in enumerate(zip(left.prompt, right.prompt)) if a != b
        )
        self.assertLess(divergence_index, 130)

        self.assertIn("three-quarter view", left.negative_prompt)
        self.assertIn("three-quarter view", right.negative_prompt)
        self.assertIn("right ear visible", left.negative_prompt)
        self.assertIn("nose pointing toward the RIGHT edge of the frame", left.negative_prompt)
        self.assertIn("left ear visible", right.negative_prompt)
        self.assertIn("nose pointing toward the LEFT edge of the frame", right.negative_prompt)
        self.assertNotEqual(left.negative_prompt, right.negative_prompt)

    def test_left_and_right_v6_profiles_place_asymmetric_hardware_on_the_correct_side(self):
        # Coordinator's own inspection of the live final-gate set: the
        # source's asymmetric ear hardware and glowing neck marking (on the
        # character's own LEFT side) appeared in the RIGHT-labeled output
        # while LEFT showed a plainer ear -- duplicated/swapped
        # side-of-identity evidence harms multi-view 3D reconstruction. The
        # rule must be generic (usable for any reference, not hardcoded to
        # this alien source): each profile conditionally reproduces the
        # hardware only when the reference shows it on the matching side,
        # and otherwise renders a plain, unmodified ear/neck. v6 measured
        # this against zix.webp across 12 seeds (6 per side): the SHOW
        # branch held 5/6 of the time on the hardware side, but the
        # SUPPRESS branch leaked an invented device (a neck collar, a
        # wired implant, an in-ear cable) on 4/6 of the plain-side
        # renders -- v6 widens the suppression clause from "a plain,
        # unmodified ear" to an exhaustive list covering the whole head
        # and neck, with matching negative terms for the concrete leak
        # shapes observed (collar, implant, in-ear cable).
        left = resolve_view_synthesis_profile("left")
        right = resolve_view_synthesis_profile("right")

        for profile, near_side, far_side in ((left, "LEFT", "RIGHT"), (right, "RIGHT", "LEFT")):
            with self.subTest(profile=profile.id):
                self.assertIn(
                    f"it sits on the character's {near_side} side, this view MUST show it",
                    profile.prompt,
                )
                self.assertIn(
                    f"hardware sits on the character's {far_side} side in the reference, "
                    "this",
                    profile.prompt,
                )
                self.assertIn("no device, no implant, no cable, no wire", profile.prompt)
                self.assertIn("hardware on the wrong side", profile.negative_prompt)
                self.assertIn("hardware duplicated on both ears", profile.negative_prompt)
                self.assertIn("symmetric hardware on both sides", profile.negative_prompt)
                self.assertIn("implant collar", profile.negative_prompt)
                self.assertIn("in-ear device", profile.negative_prompt)

    def test_left_and_right_v6_profiles_also_chain_head_crop_reference(self):
        # Escalated per the back-view v4 precedent (bead ncxzd's mechanism):
        # left/right now also get the close-up head/shoulders ReferenceLatent
        # so the model has a hard-to-ignore anchor for the exact ear/neck
        # region whose asymmetry it otherwise has to infer from the
        # full-body reference alone.
        left = resolve_view_synthesis_profile("left")
        right = resolve_view_synthesis_profile("right")

        self.assertTrue(left.head_crop_reference)
        self.assertTrue(right.head_crop_reference)

    def test_left_and_right_v4_profiles_hold_the_far_arm_visible_in_a_t_pose(self):
        # Follow-up: a strict side profile forces one T-pose arm to
        # foreshorten toward the camera and the other to extend away from
        # it, and an early render of the hardened v4 RIGHT profile against a
        # genuine T-pose reference showed the far arm fully disappearing
        # behind the torso instead of holding the pose -- not "arms at
        # sides" (the near arm was clearly extended), but a real usability
        # gap for a rigging reference. The pose clause must say the far arm
        # stays visible past the torso silhouette, and the negative prompt
        # must reject both a relaxed T-pose and a single-visible-arm result.
        left = resolve_view_synthesis_profile("left")
        right = resolve_view_synthesis_profile("right")

        for profile in (left, right):
            with self.subTest(profile=profile.id):
                self.assertIn("foreshortened toward the viewer", profile.prompt)
                # v7 purged the word "silhouette" from the prompts (it was
                # teaching the model to render literal silhouettes, bead
                # idn1d) -- the far-arm clause now says "outline".
                self.assertIn(
                    "still visible past the outline of the torso, not hidden or "
                    "cropped by the body",
                    profile.prompt,
                )
                self.assertIn("T-pose never relaxed", profile.prompt)
                self.assertIn("arm hidden behind the torso", profile.negative_prompt)
                self.assertIn("single arm visible", profile.negative_prompt)
                self.assertIn("T-pose relaxed", profile.negative_prompt)

    def test_view_synthesis_profiles_lock_the_reference_pose_exactly(self):
        # Bead t0z17: v1's "standing in the same pose as the reference" was
        # loose enough that Flux.2 Klein relaxed a T-pose front reference to
        # arms-at-sides on every synthesized view. v2 must say, in each
        # profile's prompt, that the exact arm/elbow/leg/hand geometry is
        # held -- including the literal T-pose case -- and must reject pose
        # drift in the negative prompt.
        expected_id_suffix = {"back": "-v4", "left": "-v7", "right": "-v7"}
        for view in ("back", "left", "right"):
            profile = resolve_view_synthesis_profile(view)
            with self.subTest(view=view):
                self.assertIn(expected_id_suffix[view], profile.id)
                self.assertIn("exact same body pose", profile.prompt)
                self.assertIn("arm angle, elbow angle, leg stance, and hand position exactly", profile.prompt)
                self.assertIn("T-pose", profile.prompt)
                self.assertIn("straight out horizontally", profile.prompt)
                self.assertIn("changed pose", profile.negative_prompt)
                self.assertIn("arms at sides", profile.negative_prompt)
                self.assertIn("arms lowered", profile.negative_prompt)

    def test_view_synthesis_profiles_reject_hallucinated_identity(self):
        # P1 image-adherence bug: on an out-of-distribution source (a bald
        # cyborg alien) the back-view redraw discarded the reference almost
        # entirely and fell back to a generic person with long hair -- the
        # v2 "same character" instruction was too loose to stop the model
        # from filling in unseen geometry (the back of the head) from its
        # training prior instead of the reference. v3 must name the specific
        # traits that have to survive and explicitly forbid inventing a
        # hairstyle the reference does not show.
        for view in ("back", "left", "right"):
            profile = resolve_view_synthesis_profile(view)
            with self.subTest(view=view):
                self.assertIn("recognizably the same individual", profile.prompt)
                self.assertIn("skin tone and skin markings", profile.prompt)
                self.assertIn("bald or hairless", profile.prompt)
                self.assertIn("no hair added", profile.prompt)
                self.assertIn("invented hairstyle", profile.negative_prompt)
                self.assertIn("hair added when the reference is bald", profile.negative_prompt)
                self.assertIn("generic person", profile.negative_prompt)

    def test_back_v4_profile_makes_baldness_unconditional_and_names_scalp_circuitry(self):
        # P1: on a bald cyborg alien source (skull circuitry, no hair at all),
        # v3's back view discarded the reference and rendered a generic
        # long-haired elf -- v3's bald-headed instruction was a conditional
        # aside inside a longer identity sentence, easy for the model to drop
        # when it has to invent the one region (the rear of the skull) the
        # reference never shows. v4 must front-load bald-headedness as an
        # unconditional, literal instruction for this view and name the
        # circuitry continuing onto the back of the scalp/neck as concrete
        # evidence of bare skin, not hair -- plus reject the specific hair
        # shapes (long hair, ponytail, braid, wig, hood up) that showed up in
        # the failing render, not just the generic "invented hairstyle" term.
        profile = resolve_view_synthesis_profile("back")

        self.assertEqual(profile.id, "view-synthesis-back-v4")
        self.assertEqual(profile.version, 4)
        self.assertIn("BALD head with bare scalp visible", profile.prompt)
        self.assertIn("continue onto the rear of the scalp", profile.prompt)
        self.assertIn("down the back of the neck", profile.prompt)
        self.assertIn("no hair added", profile.prompt)
        self.assertIn("long hair", profile.negative_prompt)
        self.assertIn("hairstyle", profile.negative_prompt)
        self.assertIn("wig", profile.negative_prompt)
        self.assertIn("ponytail", profile.negative_prompt)
        self.assertIn("braid", profile.negative_prompt)
        self.assertIn("hood up", profile.negative_prompt)
        self.assertIn("different character", profile.negative_prompt)
        self.assertIn("circuitry not continuing onto the back of the head", profile.negative_prompt)

    def test_unknown_view_synthesis_label_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Unsupported view synthesis label"):
            resolve_view_synthesis_profile("top")

    def test_view_synthesis_metadata_shape_records_provenance(self):
        profile = resolve_view_synthesis_profile("back")
        metadata = build_view_synthesis_metadata(
            base_metadata={"stage": "synthesize-views", "settings": {}},
            comfy_result={"prompt_id": "prompt-1", "model": "flux-2-klein-base-4b-fp8.safetensors"},
            view="back",
            source_artifact_id="front-artifact-id",
            profile=profile,
            seed=7,
        )

        self.assertEqual(metadata["synthesized_view"], "back")
        self.assertEqual(metadata["source_artifact_id"], "front-artifact-id")
        self.assertEqual(metadata["adapter"], "comfyui")
        self.assertEqual(metadata["seed"], 7)
        self.assertEqual(metadata["view_synthesis_profile"]["id"], "view-synthesis-back-v4")
        # The comfy result and the caller's base metadata both survive untouched.
        self.assertEqual(metadata["prompt_id"], "prompt-1")
        self.assertEqual(metadata["stage"], "synthesize-views")

    def test_synthesize_back_view_uploads_runs_and_downloads_a_redrawn_image(self):
        class Response:
            def __init__(self, payload=None, content=b""):
                self.payload = payload
                self.content = content

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        class Session:
            def __init__(self):
                self.posts = []

            def post(self, url, **kwargs):
                self.posts.append((url, kwargs))
                if url.endswith("/upload/image"):
                    return Response({"name": "mary-front-upload.png"})
                return Response({"prompt_id": "prompt-back", "node_errors": {}})

            def get(self, url, **kwargs):
                if "/history/" in url:
                    return Response({"prompt-back": {
                        "status": {"completed": True, "status_str": "success"},
                        "outputs": {"91": {"images": [{
                            "filename": "back.png", "subfolder": "aismith", "type": "output",
                        }]}},
                    }})
                return Response(content=b"synthesized-back-png")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mary-front.png"
            output = root / "back.png"
            source.write_bytes(_synthetic_reference_png())
            session = Session()

            metadata = run_comfyui_image_edit(
                source,
                output,
                profile=resolve_view_synthesis_profile("back"),
                seed=1,
                base_url="http://comfy.test",
                session=session,
            )

            self.assertEqual(output.read_bytes(), b"synthesized-back-png")

        self.assertEqual(metadata["prompt_id"], "prompt-back")
        submitted = [post for post in session.posts if post[0].endswith("/prompt")][0][1]["json"]["prompt"]
        encoded_prompts = [
            node["inputs"]["text"]
            for node in submitted.values()
            if node["class_type"] == "CLIPTextEncode"
        ]
        self.assertTrue(any("directly behind" in prompt for prompt in encoded_prompts))

    def test_back_v4_workflow_chains_head_crop_reference_symmetrically(self):
        # Escalation (bead ncxzd's mechanism): the back profile's
        # head_crop_reference flag must wire a SECOND ReferenceLatent pair
        # onto both positive and negative conditioning, exactly like the
        # pose-guide chain, so CFG's differential stays pointed at the text
        # instruction instead of amplifying the crop latent unopposed.
        profile = resolve_view_synthesis_profile("back")
        self.assertTrue(profile.head_crop_reference)

        workflow = build_flux2_klein_image_edit(
            input_name="zix-source.png",
            head_crop_reference_name="zix-head-crop.png",
            output_prefix="aismith/zix-back",
            profile=profile,
            seed=201,
        )

        self.assertEqual(workflow["97"]["inputs"]["image"], "zix-head-crop.png")
        self.assertEqual(workflow["100"]["class_type"], "ReferenceLatent")
        self.assertEqual(workflow["100"]["inputs"]["conditioning"], ["76", 0])
        self.assertEqual(workflow["100"]["inputs"]["latent"], ["98", 0])
        self.assertEqual(workflow["101"]["class_type"], "ReferenceLatent")
        self.assertEqual(workflow["101"]["inputs"]["conditioning"], ["77", 0])
        self.assertEqual(workflow["101"]["inputs"]["latent"], ["98", 0])
        self.assertEqual(workflow["63"]["inputs"]["positive"], ["100", 0])
        self.assertEqual(workflow["63"]["inputs"]["negative"], ["101", 0])

    def test_head_crop_reference_stacks_after_a_pose_guide_when_both_present(self):
        # Defensive: if a future profile ever combines a pose guide AND a
        # head-crop reference, the head-crop pair must chain onto whatever
        # the pose guide already wired in (92/96), not clobber back onto the
        # base identity latent (76/77) -- verifies the chaining is a stack,
        # not two independent overwrites of node 63's inputs.
        profile = ImagePreparationProfile(
            id="test-pose-and-crop",
            version=1,
            prompt="test",
            negative_prompt="test",
            pose_guide_asset="assets/pose-guides/humanoid-t-pose-v2.png",
            head_crop_reference=True,
        )

        workflow = build_flux2_klein_image_edit(
            input_name="source.png",
            pose_input_name="pose-guide.png",
            head_crop_reference_name="head-crop.png",
            output_prefix="aismith/combo",
            profile=profile,
            seed=1,
        )

        self.assertEqual(workflow["100"]["inputs"]["conditioning"], ["92", 0])
        self.assertEqual(workflow["101"]["inputs"]["conditioning"], ["96", 0])
        self.assertEqual(workflow["63"]["inputs"]["positive"], ["100", 0])
        self.assertEqual(workflow["63"]["inputs"]["negative"], ["101", 0])

    def test_run_comfyui_image_edit_uploads_and_wires_the_head_crop_reference(self):
        class Response:
            def __init__(self, payload=None, content=b""):
                self.payload = payload
                self.content = content

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        class Session:
            def __init__(self):
                self.posts = []
                self.upload_count = 0

            def post(self, url, **kwargs):
                self.posts.append((url, kwargs))
                if url.endswith("/upload/image"):
                    self.upload_count += 1
                    return Response({"name": f"upload-{self.upload_count}.png"})
                return Response({"prompt_id": "prompt-back-crop", "node_errors": {}})

            def get(self, url, **kwargs):
                if "/history/" in url:
                    return Response({"prompt-back-crop": {
                        "status": {"completed": True, "status_str": "success"},
                        "outputs": {"91": {"images": [{
                            "filename": "back.png", "subfolder": "aismith", "type": "output",
                        }]}},
                    }})
                return Response(content=b"synthesized-back-png")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "zix-front.png"
            output = root / "back.png"
            source.write_bytes(_synthetic_reference_png())
            session = Session()

            metadata = run_comfyui_image_edit(
                source,
                output,
                profile=resolve_view_synthesis_profile("back"),
                seed=201,
                base_url="http://comfy.test",
                session=session,
            )

        uploads = [post for post in session.posts if post[0].endswith("/upload/image")]
        self.assertEqual(len(uploads), 2, "source image and head-crop reference must both upload")
        submitted = [post for post in session.posts if post[0].endswith("/prompt")][0][1]["json"]["prompt"]
        self.assertEqual(submitted["90"]["inputs"]["image"], "upload-1.png")
        self.assertEqual(submitted["97"]["inputs"]["image"], "upload-2.png")
        self.assertTrue(metadata["head_crop_reference"])
        self.assertRegex(metadata["head_crop_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(metadata["workflow_version"], "flux2-klein-multi-reference-image-edit-v2")


class SourcePoseViewSynthesisTests(unittest.TestCase):
    """The `source-pose` family redraws the SOURCE's pose (arms/held object
    as photographed) instead of re-imposing a T-pose -- the tpose family's
    prompts actively demand T-pose arm geometry and drop held objects when
    run against a non-T-pose photo (2026-09-05 live run)."""

    def test_default_family_is_unchanged(self):
        self.assertIs(resolve_view_synthesis_profile("back"), VIEW_SYNTHESIS_PROFILES["back"])

    def test_source_pose_family_resolves_the_new_profiles(self):
        for view in ("back", "left", "right"):
            with self.subTest(view=view):
                profile = resolve_view_synthesis_profile(view, family="source-pose")
                self.assertIs(profile, SOURCE_POSE_VIEW_SYNTHESIS_PROFILES[view])
                self.assertEqual(profile.version, 1)
                self.assertTrue(profile.head_crop_reference)

    def test_source_pose_profile_ids(self):
        self.assertEqual(
            {view: profile.id for view, profile in SOURCE_POSE_VIEW_SYNTHESIS_PROFILES.items()},
            {
                "back": "source-pose-view-back-v1",
                "left": "source-pose-view-left-v1",
                "right": "source-pose-view-right-v1",
            },
        )

    def test_unknown_family_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Unsupported view synthesis profile family"):
            resolve_view_synthesis_profile("back", family="nope")

    def test_source_pose_profiles_hold_the_source_pose_and_held_object(self):
        for view in ("back", "left", "right"):
            with self.subTest(view=view):
                profile = SOURCE_POSE_VIEW_SYNTHESIS_PROFILES[view]
                self.assertIn("same object is held", profile.prompt)
                self.assertIn("T-pose", profile.negative_prompt)
                self.assertNotIn("arms lowered", profile.negative_prompt)

    def test_tpose_profiles_still_enforce_t_pose_regression_guard(self):
        for view in ("back", "left", "right"):
            with self.subTest(view=view):
                profile = VIEW_SYNTHESIS_PROFILES[view]
                self.assertIn("arms lowered", profile.negative_prompt)

    def test_profile_families_table_exposes_both_families(self):
        # mesh-guided (a third family) and mesh-guided-idfirst (a fourth) are
        # added and tested separately in MeshGuidedProfileTests; this guards
        # the two pre-existing entries.
        self.assertEqual(
            set(VIEW_SYNTHESIS_PROFILE_FAMILIES),
            {"tpose", "source-pose", "mesh-guided", "mesh-guided-plain", "mesh-guided-idfirst"},
        )
        self.assertIs(VIEW_SYNTHESIS_PROFILE_FAMILIES["tpose"], VIEW_SYNTHESIS_PROFILES)
        self.assertIs(VIEW_SYNTHESIS_PROFILE_FAMILIES["source-pose"], SOURCE_POSE_VIEW_SYNTHESIS_PROFILES)


class HeadCropReferenceTests(unittest.TestCase):
    def test_detect_subject_bbox_locates_a_solid_subject_against_a_uniform_background(self):
        png_bytes = _synthetic_reference_png(
            size=(100, 100), background=(200, 200, 200), subject_box=(30, 10, 70, 90), subject=(20, 20, 20),
        )

        left, top, right, bottom = detect_subject_bbox(png_bytes)

        self.assertEqual((left, top, right, bottom), (30, 10, 70, 90))

    def test_detect_subject_bbox_falls_back_to_the_full_frame_when_nothing_stands_out(self):
        blank = Image.new("RGB", (50, 50), (128, 128, 128))
        buffer = BytesIO()
        blank.save(buffer, format="PNG")

        left, top, right, bottom = detect_subject_bbox(buffer.getvalue())

        self.assertEqual((left, top, right, bottom), (0, 0, 50, 50))

    def test_build_head_crop_reference_crops_the_top_fraction_of_the_bbox(self):
        png_bytes = _synthetic_reference_png(
            size=(100, 100), background=(200, 200, 200), subject_box=(30, 10, 70, 90), subject=(20, 20, 20),
        )

        cropped_bytes = build_head_crop_reference(png_bytes, crop_fraction=0.35)

        with Image.open(BytesIO(cropped_bytes)) as cropped:
            # bbox height is 80px (10..90); 35% of that is 28px.
            self.assertEqual(cropped.size, (40, 28))
            # The crop should be solid "subject" color throughout -- it's
            # entirely inside the subject box, not the background.
            self.assertEqual(cropped.convert("RGB").getpixel((5, 5)), (20, 20, 20))


class SourcePoseViewsDriverTests(unittest.TestCase):
    """scripts/source_pose_views.py drives the exact same call path as the
    `synthesize-views` job (`_synthesize_view_into`), headlessly."""

    def test_driver_calls_run_comfyui_image_edit_once_per_view_and_writes_manifest(self):
        import scripts.source_pose_views as source_pose_views

        calls = []

        def fake_edit(source, destination, *, profile, seed, base_url, pose_guide_root=None):
            calls.append({
                "source": source, "destination": destination, "profile": profile,
                "seed": seed, "base_url": base_url, "pose_guide_root": pose_guide_root,
            })
            Path(destination).write_bytes(b"png-bytes")
            return {"prompt_id": f"prompt-{profile.id}", "model": "test"}

        with TemporaryDirectory() as directory:
            photo = Path(directory) / "photo.png"
            photo.write_bytes(b"source-bytes")
            out_dir = Path(directory) / "out"

            with patch.object(source_pose_views, "run_comfyui_image_edit", fake_edit):
                import sys
                argv = [
                    "source_pose_views.py",
                    "--photo", str(photo),
                    "--out", str(out_dir),
                    "--views", "back,left,right",
                    "--seed", "7",
                    "--family", "source-pose",
                    "--comfyui", "http://127.0.0.1:8188",
                ]
                with patch.object(sys, "argv", argv):
                    source_pose_views.main()

            self.assertEqual(len(calls), 3)
            self.assertEqual([call["seed"] for call in calls], [8, 9, 10])
            self.assertEqual(
                {call["profile"].id for call in calls},
                {"source-pose-view-back-v1", "source-pose-view-left-v1", "source-pose-view-right-v1"},
            )
            for call in calls:
                self.assertEqual(call["base_url"], "http://127.0.0.1:8188")
                self.assertIsNotNone(call["pose_guide_root"])

            manifest = json.loads((out_dir / "views.json").read_text())
            self.assertEqual(manifest["seed"], 7)
            self.assertEqual(manifest["family"], "source-pose")
            self.assertEqual(
                manifest["profile_ids"],
                {
                    "back": "source-pose-view-back-v1",
                    "left": "source-pose-view-left-v1",
                    "right": "source-pose-view-right-v1",
                },
            )
            for view in ("back", "left", "right"):
                self.assertTrue(Path(manifest[view]).is_file())


class MeshGuidedGraphTests(unittest.TestCase):
    """Mesh-guided views paint a clay render as the character: identity comes
    from a separate photo reference chained like the pose guide, on top of
    the source (the clay render itself)."""

    def test_identity_reference_chains_source_then_identity_then_head_crop(self):
        profile = MESH_GUIDED_VIEW_SYNTHESIS_PROFILES["front"]
        self.assertTrue(profile.identity_reference)
        self.assertTrue(profile.head_crop_reference)

        workflow = build_flux2_klein_image_edit(
            input_name="clay-front.png",
            identity_reference_name="photo-identity.png",
            head_crop_reference_name="photo-head-crop.png",
            output_prefix="aismith/mesh-front",
            profile=profile,
            seed=7,
        )

        load_images = {
            node["inputs"]["image"]
            for node in workflow.values()
            if node["class_type"] == "LoadImage"
        }
        self.assertEqual(
            load_images, {"clay-front.png", "photo-identity.png", "photo-head-crop.png"},
        )

        # source (76/77) -> identity (105/106), symmetric on both sides.
        self.assertEqual(workflow["105"]["class_type"], "ReferenceLatent")
        self.assertEqual(workflow["105"]["inputs"]["conditioning"], ["76", 0])
        self.assertEqual(workflow["106"]["class_type"], "ReferenceLatent")
        self.assertEqual(workflow["106"]["inputs"]["conditioning"], ["77", 0])

        # identity (105/106) -> head crop (100/101), symmetric on both sides.
        self.assertEqual(workflow["100"]["inputs"]["conditioning"], ["105", 0])
        self.assertEqual(workflow["101"]["inputs"]["conditioning"], ["106", 0])

        self.assertEqual(workflow["63"]["inputs"]["positive"], ["100", 0])
        self.assertEqual(workflow["63"]["inputs"]["negative"], ["101", 0])

    def test_missing_identity_image_fails_closed(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "clay-front.png"
            source.write_bytes(b"clay-bytes")
            output = Path(directory) / "front.png"

            with self.assertRaisesRegex(ValueError, "identity_reference=True"):
                run_comfyui_image_edit(
                    source,
                    output,
                    profile=MESH_GUIDED_VIEW_SYNTHESIS_PROFILES["front"],
                    seed=1,
                    base_url="http://comfy.test",
                    session=None,
                )


class MeshGuidedProfileTests(unittest.TestCase):
    ROLES = (
        "front", "back", "left", "right",
        "front-left", "front-right", "back-left", "back-right",
    )

    def test_all_eight_roles_resolve_for_the_mesh_guided_family(self):
        for role in self.ROLES:
            with self.subTest(role=role):
                profile = resolve_view_synthesis_profile(role, family="mesh-guided")
                self.assertIs(profile, MESH_GUIDED_VIEW_SYNTHESIS_PROFILES[role])
                self.assertEqual(profile.id, f"mesh-guided-view-{role}-v1")
                self.assertTrue(profile.identity_reference)

    def test_every_prompt_names_the_clay_render_and_pixel_fidelity(self):
        for role in self.ROLES:
            with self.subTest(role=role):
                profile = MESH_GUIDED_VIEW_SYNTHESIS_PROFILES[role]
                self.assertIn("clay render", profile.prompt)
                self.assertIn("pixel for pixel", profile.prompt)

    def test_every_negative_forbids_a_changed_silhouette(self):
        for role in self.ROLES:
            with self.subTest(role=role):
                profile = MESH_GUIDED_VIEW_SYNTHESIS_PROFILES[role]
                self.assertIn("changed silhouette", profile.negative_prompt)

    def test_back_roles_reuse_the_bald_scalp_hardening_sentence(self):
        for role in ("back", "back-left", "back-right"):
            with self.subTest(role=role):
                profile = MESH_GUIDED_VIEW_SYNTHESIS_PROFILES[role]
                self.assertIn(
                    "the back of the head in this view is a BALD head with bare scalp",
                    profile.prompt,
                )

    def test_head_crop_reference_only_on_the_roles_the_photo_actually_covers(self):
        # The head crop is built from the FRONT photo -- useful for views
        # that still show some of the front of the head, not for the three
        # rear views, which instead rely on the bald-scalp prompt hardening.
        for role in ("front", "left", "right", "front-left", "front-right"):
            with self.subTest(role=role):
                self.assertTrue(MESH_GUIDED_VIEW_SYNTHESIS_PROFILES[role].head_crop_reference)
        for role in ("back", "back-left", "back-right"):
            with self.subTest(role=role):
                self.assertFalse(MESH_GUIDED_VIEW_SYNTHESIS_PROFILES[role].head_crop_reference)

    def test_every_mesh_guided_prompt_starts_with_the_indexed_image_sentence(self):
        # Lever 1 (indexed prompting): the images have no inherent labels, so
        # every mesh-guided prompt must open by naming them by position.
        for role in self.ROLES:
            with self.subTest(role=role):
                profile = MESH_GUIDED_VIEW_SYNTHESIS_PROFILES[role]
                self.assertTrue(
                    profile.prompt.startswith(
                        "The first image is an untextured grey clay render"
                    )
                )
                self.assertIn(
                    "The second image is the reference photo of the character to paint.",
                    profile.prompt,
                )
                if profile.head_crop_reference:
                    self.assertIn(
                        "The third image is a close-up of the same character's head.",
                        profile.prompt,
                    )
                else:
                    self.assertNotIn("The third image", profile.prompt)

    def test_tpose_and_source_pose_families_are_unchanged(self):
        self.assertEqual(
            set(VIEW_SYNTHESIS_PROFILE_FAMILIES),
            {"tpose", "source-pose", "mesh-guided", "mesh-guided-plain", "mesh-guided-idfirst"},
        )
        self.assertIs(VIEW_SYNTHESIS_PROFILE_FAMILIES["tpose"], VIEW_SYNTHESIS_PROFILES)
        self.assertIs(VIEW_SYNTHESIS_PROFILE_FAMILIES["source-pose"], SOURCE_POSE_VIEW_SYNTHESIS_PROFILES)
        self.assertIs(VIEW_SYNTHESIS_PROFILE_FAMILIES["mesh-guided"], MESH_GUIDED_VIEW_SYNTHESIS_PROFILES)
        for view in ("back", "left", "right"):
            self.assertIs(resolve_view_synthesis_profile(view), VIEW_SYNTHESIS_PROFILES[view])
            self.assertIs(
                resolve_view_synthesis_profile(view, family="source-pose"),
                SOURCE_POSE_VIEW_SYNTHESIS_PROFILES[view],
            )


class MeshGuidedPlainProfileTests(unittest.TestCase):
    """The plain family is the V0 wording: the indexed 'first image / second
    image' intro removed (it cost the front view 0.73 -> 0.16 SSIM), and
    everything else -- references, negatives, per-view sentence -- identical."""

    def test_plain_family_strips_only_the_indexed_intro(self):
        self.assertEqual(set(MESH_GUIDED_PLAIN_VIEW_SYNTHESIS_PROFILES), set(MESH_GUIDED_VIEW_SYNTHESIS_PROFILES))
        for view, indexed in MESH_GUIDED_VIEW_SYNTHESIS_PROFILES.items():
            plain = MESH_GUIDED_PLAIN_VIEW_SYNTHESIS_PROFILES[view]
            self.assertEqual(plain.id, indexed.id.replace("mesh-guided-", "mesh-guided-plain-", 1))
            self.assertFalse(plain.prompt.startswith("The first image"))
            self.assertTrue(indexed.prompt.endswith(plain.prompt))
            self.assertTrue(plain.prompt.startswith("This is the character"))
            self.assertEqual(plain.negative_prompt, indexed.negative_prompt)
            self.assertEqual(plain.identity_reference, indexed.identity_reference)
            self.assertEqual(plain.head_crop_reference, indexed.head_crop_reference)
            self.assertEqual(plain.identity_first, indexed.identity_first)

    def test_plain_family_resolves_by_name(self):
        self.assertIs(
            resolve_view_synthesis_profile("front", family="mesh-guided-plain"),
            MESH_GUIDED_PLAIN_VIEW_SYNTHESIS_PROFILES["front"],
        )


class MeshGuidedIdentityFirstTests(unittest.TestCase):
    """Lever 3: the mesh-guided-idfirst family chains the identity photo onto
    the base conditioning BEFORE the clay/source latent."""

    ROLES = (
        "front", "back", "left", "right",
        "front-left", "front-right", "back-left", "back-right",
    )

    def test_all_eight_roles_resolve_for_the_idfirst_family(self):
        for role in self.ROLES:
            with self.subTest(role=role):
                profile = resolve_view_synthesis_profile(role, family="mesh-guided-idfirst")
                self.assertEqual(profile.id, f"mesh-guided-idfirst-view-{role}-v1")
                self.assertTrue(profile.identity_reference)
                self.assertTrue(profile.identity_first)

    def test_every_idfirst_prompt_starts_with_the_reference_photo_sentence(self):
        for role in self.ROLES:
            with self.subTest(role=role):
                profile = resolve_view_synthesis_profile(role, family="mesh-guided-idfirst")
                self.assertTrue(
                    profile.prompt.startswith("The first image is the reference photo")
                )

    def test_identity_first_chains_identity_then_source_then_head_crop(self):
        profile = resolve_view_synthesis_profile("front", family="mesh-guided-idfirst")
        self.assertTrue(profile.identity_first)
        self.assertTrue(profile.head_crop_reference)

        workflow = build_flux2_klein_image_edit(
            input_name="clay-front.png",
            identity_reference_name="photo-identity.png",
            head_crop_reference_name="photo-head-crop.png",
            output_prefix="aismith/mesh-idfirst-front",
            profile=profile,
            seed=7,
            identity_first=profile.identity_first,
        )

        # Follow the conditioning links from the base CLIPTextEncode nodes
        # (74 positive, 67 negative) forward through the ReferenceLatent
        # chain to confirm identity is attached FIRST and source SECOND.
        by_type = {}
        for node_id, node in workflow.items():
            by_type.setdefault(node["class_type"], []).append(node_id)

        reference_latents = {
            node_id: node
            for node_id, node in workflow.items()
            if node["class_type"] == "ReferenceLatent"
        }
        # Identity (104) must chain directly onto the base text conditioning.
        identity_positive = next(
            node_id
            for node_id, node in reference_latents.items()
            if node["inputs"]["latent"] == ["104", 0]
            and node["inputs"]["conditioning"] == ["74", 0]
        )
        identity_negative = next(
            node_id
            for node_id, node in reference_latents.items()
            if node["inputs"]["latent"] == ["104", 0]
            and node["inputs"]["conditioning"] == ["67", 0]
        )
        # Source (78) must chain onto the identity's own ReferenceLatent
        # nodes, not onto the base text conditioning directly.
        source_positive = next(
            node_id
            for node_id, node in reference_latents.items()
            if node["inputs"]["latent"] == ["78", 0]
            and node["inputs"]["conditioning"] == [identity_positive, 0]
        )
        source_negative = next(
            node_id
            for node_id, node in reference_latents.items()
            if node["inputs"]["latent"] == ["78", 0]
            and node["inputs"]["conditioning"] == [identity_negative, 0]
        )
        # Head crop (98) must chain onto the source's ReferenceLatent nodes,
        # last in the stack, and be the final input into the guider.
        self.assertEqual(
            workflow["63"]["inputs"]["positive"][0],
            next(
                node_id
                for node_id, node in reference_latents.items()
                if node["inputs"]["latent"] == ["98", 0]
                and node["inputs"]["conditioning"] == [source_positive, 0]
            ),
        )
        self.assertEqual(
            workflow["63"]["inputs"]["negative"][0],
            next(
                node_id
                for node_id, node in reference_latents.items()
                if node["inputs"]["latent"] == ["98", 0]
                and node["inputs"]["conditioning"] == [source_negative, 0]
            ),
        )

    def test_identity_first_false_keeps_source_first_for_the_plain_mesh_guided_family(self):
        # Regression guard: identity_first defaults to False, and the plain
        # mesh-guided family (identity_first not set) still chains source
        # (76/77) before identity (105/106), as before this task.
        profile = MESH_GUIDED_VIEW_SYNTHESIS_PROFILES["front"]
        self.assertFalse(profile.identity_first)

        workflow = build_flux2_klein_image_edit(
            input_name="clay-front.png",
            identity_reference_name="photo-identity.png",
            output_prefix="aismith/mesh-front",
            profile=profile,
            seed=7,
            identity_first=profile.identity_first,
        )

        self.assertEqual(workflow["76"]["inputs"]["latent"], ["78", 0])
        self.assertEqual(workflow["105"]["inputs"]["conditioning"], ["76", 0])
        self.assertEqual(workflow["105"]["inputs"]["latent"], ["104", 0])


class MeshGuidedViewsDriverTests(unittest.TestCase):
    """scripts/mesh_guided_views.py drives run_comfyui_image_edit once per
    role, feeding it the pre-rendered clay guide as `source` and the photo
    as `identity_image`."""

    ROLES = (
        "front", "back", "left", "right",
        "front-left", "front-right", "back-left", "back-right",
    )

    def test_driver_calls_run_comfyui_image_edit_once_per_role_and_writes_manifest(self):
        import scripts.mesh_guided_views as mesh_guided_views

        calls = []

        def fake_edit(
            source, destination, *, profile, seed, base_url,
            identity_image=None, pose_guide_root=None,
        ):
            calls.append({
                "source": source, "destination": destination, "profile": profile,
                "seed": seed, "base_url": base_url, "identity_image": identity_image,
                "pose_guide_root": pose_guide_root,
            })
            Path(destination).write_bytes(b"png-bytes")
            return {"prompt_id": f"prompt-{profile.id}", "model": "test"}

        with TemporaryDirectory() as directory:
            photo = Path(directory) / "photo.png"
            photo.write_bytes(b"photo-bytes")
            guides_dir = Path(directory) / "guides"
            guides_dir.mkdir()
            roles_manifest = {}
            for role in self.ROLES:
                clay_path = guides_dir / f"{role}-clay.png"
                clay_path.write_bytes(b"clay-bytes")
                roles_manifest[role] = {"clay": str(clay_path)}
            (guides_dir / "guides.json").write_text(json.dumps({"roles": roles_manifest}))
            out_dir = Path(directory) / "out"

            with patch.object(mesh_guided_views, "run_comfyui_image_edit", fake_edit):
                import sys
                argv = [
                    "mesh_guided_views.py",
                    "--guides", str(guides_dir),
                    "--photo", str(photo),
                    "--out", str(out_dir),
                    "--seed", "7",
                    "--comfyui", "http://127.0.0.1:8188",
                ]
                with patch.object(sys, "argv", argv):
                    mesh_guided_views.main()

            self.assertEqual(len(calls), 8)
            self.assertEqual([call["seed"] for call in calls], list(range(8, 16)))
            self.assertEqual(
                {call["profile"].id for call in calls},
                {f"mesh-guided-view-{role}-v1" for role in self.ROLES},
            )
            for call in calls:
                self.assertEqual(call["base_url"], "http://127.0.0.1:8188")
                self.assertEqual(call["identity_image"], photo)
                self.assertIsNotNone(call["pose_guide_root"])

            self.assertTrue((out_dir / "guides.json").is_file())
            manifest = json.loads((out_dir / "views.json").read_text())
            self.assertEqual(manifest["seed"], 7)
            self.assertEqual(manifest["family"], "mesh-guided")
            self.assertEqual(
                manifest["profile_ids"],
                {role: f"mesh-guided-view-{role}-v1" for role in self.ROLES},
            )
            for role in self.ROLES:
                self.assertTrue(Path(manifest[role]).is_file())

    def test_identity_map_sends_the_matched_role_photo_with_fallback(self):
        # Lever 2: a role listed in --identity-map uses that image as the
        # identity reference; front-left/front-right and back-left/back-right
        # fall back to the nearest listed full role (front/back) unless the
        # 3/4 role itself is listed; everything else falls back to --photo.
        import scripts.mesh_guided_views as mesh_guided_views

        calls = []

        def fake_edit(
            source, destination, *, profile, seed, base_url,
            identity_image=None, pose_guide_root=None,
        ):
            calls.append({"role": profile.id, "identity_image": identity_image})
            Path(destination).write_bytes(b"png-bytes")
            return {"prompt_id": f"prompt-{profile.id}", "model": "test"}

        with TemporaryDirectory() as directory:
            photo = Path(directory) / "photo.png"
            photo.write_bytes(b"photo-bytes")
            back_photo = Path(directory) / "back.png"
            back_photo.write_bytes(b"back-photo-bytes")
            guides_dir = Path(directory) / "guides"
            guides_dir.mkdir()
            roles_manifest = {}
            for role in self.ROLES:
                clay_path = guides_dir / f"{role}-clay.png"
                clay_path.write_bytes(b"clay-bytes")
                roles_manifest[role] = {"clay": str(clay_path)}
            (guides_dir / "guides.json").write_text(json.dumps({"roles": roles_manifest}))
            out_dir = Path(directory) / "out"

            with patch.object(mesh_guided_views, "run_comfyui_image_edit", fake_edit):
                import sys
                argv = [
                    "mesh_guided_views.py",
                    "--guides", str(guides_dir),
                    "--photo", str(photo),
                    "--out", str(out_dir),
                    "--identity-map", f"back={back_photo}",
                    "--seed", "7",
                    "--comfyui", "http://127.0.0.1:8188",
                ]
                with patch.object(sys, "argv", argv):
                    mesh_guided_views.main()

            identity_by_role = {
                call["role"].removeprefix("mesh-guided-view-").removesuffix("-v1"): call["identity_image"]
                for call in calls
            }
            for role in ("back", "back-left", "back-right"):
                self.assertEqual(identity_by_role[role], back_photo)
            for role in ("front", "left", "right", "front-left", "front-right"):
                self.assertEqual(identity_by_role[role], photo)

    def test_family_flag_resolves_the_idfirst_profiles(self):
        import scripts.mesh_guided_views as mesh_guided_views

        calls = []

        def fake_edit(
            source, destination, *, profile, seed, base_url,
            identity_image=None, pose_guide_root=None,
        ):
            calls.append(profile.id)
            Path(destination).write_bytes(b"png-bytes")
            return {"prompt_id": f"prompt-{profile.id}", "model": "test"}

        with TemporaryDirectory() as directory:
            photo = Path(directory) / "photo.png"
            photo.write_bytes(b"photo-bytes")
            guides_dir = Path(directory) / "guides"
            guides_dir.mkdir()
            roles_manifest = {}
            for role in self.ROLES:
                clay_path = guides_dir / f"{role}-clay.png"
                clay_path.write_bytes(b"clay-bytes")
                roles_manifest[role] = {"clay": str(clay_path)}
            (guides_dir / "guides.json").write_text(json.dumps({"roles": roles_manifest}))
            out_dir = Path(directory) / "out"

            with patch.object(mesh_guided_views, "run_comfyui_image_edit", fake_edit):
                import sys
                argv = [
                    "mesh_guided_views.py",
                    "--guides", str(guides_dir),
                    "--photo", str(photo),
                    "--out", str(out_dir),
                    "--family", "mesh-guided-idfirst",
                    "--seed", "7",
                    "--comfyui", "http://127.0.0.1:8188",
                ]
                with patch.object(sys, "argv", argv):
                    mesh_guided_views.main()

            self.assertEqual(
                set(calls),
                {f"mesh-guided-idfirst-view-{role}-v1" for role in self.ROLES},
            )
            manifest = json.loads((out_dir / "views.json").read_text())
            self.assertEqual(manifest["family"], "mesh-guided-idfirst")


if __name__ == "__main__":
    unittest.main()
