"""Versioned character-image preparation profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import requests


@dataclass(frozen=True)
class ImagePreparationProfile:
    id: str
    version: int
    prompt: str
    negative_prompt: str
    width: int = 1024
    height: int = 1024
    pose_guide_asset: str | None = None
    # Bead ncxzd's mechanism, escalated to `back` view synthesis: when the
    # prompt hardening in v4 alone isn't enough (a weaker/out-of-distribution
    # source still lets the model invent hair on the back of a bald head, even
    # though the crown itself stays bald and marked), a second ReferenceLatent
    # -- a tight crop of the reference's own head/shoulders region -- is
    # chained symmetrically onto positive AND negative conditioning
    # (build_flux2_klein_image_edit's `head_crop_reference_name`, mirroring
    # the 45ac3e4 pose-guide chaining pattern) to give the model an
    # unambiguous, hard-to-ignore close-up anchor of exactly the skull region
    # it otherwise has to invent from its training prior.
    head_crop_reference: bool = False
    # Mesh-guided view synthesis: identity comes from a separate photo
    # reference (not the source clay render), chained the same way as the
    # pose guide -- build_flux2_klein_image_edit's `identity_reference_name`.
    identity_reference: bool = False
    # Identity-first mesh-guided order: chain the identity photo onto the
    # base text conditioning BEFORE the source/clay latent (instead of
    # after), so the character photo is CFG's primary edit image and the
    # clay render is the structural reference -- see the "mesh-guided-idfirst"
    # family below.
    identity_first: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


HUMANOID_A_POSE_SWORD_V1 = ImagePreparationProfile(
    id="humanoid-a-pose-sword-v1",
    version=1,
    prompt=(
        "Preserve the character's face, hair, clothing, colors, materials, and identity. "
        "Create one centered full-body humanoid on a plain neutral background, viewed from "
        "the front with the head and both feet fully visible. Use a relaxed A-pose with the "
        "arms separated from the torso, elbows slightly bent, legs separated, and knees "
        "readable. Keep the left hand open. Keep the right weapon hand closed around a short "
        "neutral cylindrical grip with no blade, guard, or ornament."
    ),
    negative_prompt=(
        "cropped head, cropped feet, duplicate person, duplicate limbs, missing limbs, fused "
        "limbs, crossed arms, arms touching torso, seated pose, action pose, scenery occlusion, "
        "text, watermark, no sword blade, no large weapon, no shield"
    ),
)

HUMANOID_A_POSE_SWORD_V2 = ImagePreparationProfile(
    id="humanoid-a-pose-sword-v2",
    version=2,
    prompt=(
        "Preserve the character's face, hair, clothing, colors, materials, and identity. "
        "Create one centered full-body humanoid on a plain neutral background, viewed from "
        "the front with the head and both feet fully visible. Use a relaxed A-pose with the "
        "arms separated from the torso, elbows slightly bent, legs separated, and knees "
        "readable. The left hand must be fully open, empty, and show a readable palm and "
        "separated fingers. Place exactly one grip in the image: a short neutral cylindrical "
        "grip held by the closed right hand only, with no blade, guard, or ornament."
    ),
    negative_prompt=(
        "cropped head, cropped feet, duplicate person, duplicate limbs, missing limbs, fused "
        "limbs, crossed arms, arms touching torso, seated pose, action pose, scenery occlusion, "
        "text, watermark, sword blade, large weapon, shield, two grips, duplicate grip, object "
        "in the left hand, closed left hand"
    ),
)

HUMANOID_A_POSE_RIG_V3 = ImagePreparationProfile(
    id="humanoid-a-pose-rig-v3",
    version=3,
    prompt=(
        "Preserve the character's face, hair, colors, materials, and recognizable identity. "
        "Create one centered full-body humanoid on a plain neutral background, viewed from "
        "the front with the head and both feet fully visible. Use a relaxed symmetrical A-pose "
        "with arms separated from the torso, elbows slightly bent, and both hands fully open "
        "and empty with readable palms and separated fingers. Use fitted layered fantasy "
        "clothing and fitted trousers that keep the hips, knees, and ankles visually readable. "
        "Separate the legs and feet. The weapon will be attached later in 3D."
    ),
    negative_prompt=(
        "cropped head, cropped feet, duplicate person, duplicate limbs, missing limbs, fused "
        "limbs, crossed arms, arms touching torso, seated pose, action pose, scenery occlusion, "
        "text, watermark, weapon, grip, sword, shield, object in either hand, closed hand, long "
        "dress, skirt, robe, cape, wide fabric crossing hips, fabric hiding knees"
    ),
)

# HUMANOID_T_POSE_RIG_V1's id string is a stable external contract (main.py's
# generate-pose default and frontend/src/workspace.ts's PreparationProfile
# union both hardcode "humanoid-t-pose-rig-v1"), so the id and the Python
# constant name stay put even though the profile content below is now on its
# second revision (see `version`).
#
# v2 (operator escalation, 2imyx-regression / bead ncxzd): v1's clothing
# instruction -- "fitted layered fantasy clothing" -- named no source, so on
# an out-of-distribution reference (a grey-skinned cyborg alien in a
# tattered black coat) the model treated "fantasy clothing" as license to
# invent an unrelated ornate navy-and-gold armor set, and separately drifted
# the alien's grey-green skin and glowing purple eyes toward a generic pink
# human face -- even though the same symmetric ReferenceLatent chaining from
# 45ac3e4 kept the pose and baldness correct. v2 borrows the view-synthesis
# v3 profiles' explicit anti-drift pattern: name the specific identity
# features that must survive (skin tone, eye color, markings, ear shape) and
# make the wardrobe instruction a closer-fitted *adaptation of the
# reference's own garment* -- same colors, same materials, same wear and
# damage -- instead of an open-ended "fantasy clothing" invitation, with
# matching negative-prompt terms for both failure modes.
#
# v3 (operator escalation, zix run): v2 fixed the invented-armor failure but
# over-corrected into the opposite one -- "fitted closer to the body" plus
# a wall of negatives against ornate/invented/redesigned outfits gave the
# model no positive anchor for what to actually render on that fitted
# silhouette, so it converged on the cheapest thing that satisfies "fitted"
# and violates none of the negatives: a plain, featureless, solid-color
# base layer. Two zix candidates in a row kept identity (face, circuitry,
# eyes) but rendered the operator's tattered dark coat, hood, neck
# collar/harness, and belt as a blank charcoal bodysuit -- the boots
# survived (named explicitly by "fabric hiding knees" staying clear of the
# lower leg) but everything above the ankle did not. v3 keeps v2's fitted,
# rig-ready silhouette requirement (still forbidding loose/flowing fabric)
# but now enumerates the specific garment pieces and surface details that
# must survive the fit-down: the coat/hood as a close-cut layered garment
# (not absent), its exact color and worn/weathered texture, its frayed
# edges and stitched seams, the neck collar/harness hardware, the belt or
# waist strap, and the boots' own texture and lacing -- framed as "a fitted
# adaptation of this coat," never "a plain base layer instead of it" --
# with matching negative-prompt terms for the blank-bodysuit failure mode.
POSE_GUIDES = Path(__file__).parent / "assets" / "pose-guides"


def t_pose_guide_asset(environ: Mapping[str, str] = os.environ) -> str:
    """The T-pose profile's pose guide, relative to this module's directory.

    `humanoid-t-pose-v2.png` is a CC0 render made for this repository (its
    PROVENANCE.md says how). It has not yet been validated against the
    image-edit model, so `assets/pose-guides/active.json` keeps the guide
    already in use (v1) active in this repository. Set
    AISMITH_T_POSE_GUIDE=humanoid-t-pose-v2.png to opt in to v2 for a
    controlled run. The public repository carries only v2, and its active.json
    names it. Either way, job metadata records the guide's path and sha256.
    """
    name = (environ.get("AISMITH_T_POSE_GUIDE") or "").strip()
    if not name:
        active = json.loads((POSE_GUIDES / "active.json").read_text(encoding="utf-8"))
        name = active["humanoid-t-pose-rig-v1"]
    if Path(name).name != name or not (POSE_GUIDES / name).is_file():
        raise ValueError(f"T-pose guide {name!r} is not a file in assets/pose-guides/")
    return f"assets/pose-guides/{name}"


HUMANOID_T_POSE_RIG_V1 = ImagePreparationProfile(
    id="humanoid-t-pose-rig-v1",
    version=3,
    prompt=(
        "Use the first image only as the identity and wardrobe reference. Preserve this "
        "exact character's face, skin tone, skin markings, eye color, hair, and recognizable "
        "identity: keep the same skin tone and skin markings, the same head and ear shape, the "
        "same distinctive features such as tattoos, cybernetics, or unusual skin texture, and "
        "the exact same eye color as the reference -- if the reference is bald or hairless, "
        "keep the head bald with no hair added. Use the second image only as the pose "
        "reference. Create one centered full-body humanoid on a plain neutral background, "
        "viewed from the front with the head and both feet fully visible. Use a strict "
        "symmetrical T-pose with both arms horizontal at shoulder height, elbows straight, "
        "wrists straight, and both hands fully open and empty with palms facing down and "
        "separated fingers. Dress the character in a fitted, rig-ready ADAPTATION of the SAME "
        "garments worn in the first reference image -- not a plain, blank, or featureless base "
        "layer instead of them. Every distinct garment piece the reference shows must still be "
        "visible, just cut closer to the body: if the reference wears a coat or hood, render "
        "that same coat and hood as a close-fitted layered garment rather than omitting it; "
        "keep its exact color, its worn and weathered fabric texture, its frayed or torn edges, "
        "and its visible stitched seams. Keep any collar, harness, or hardware worn at the neck "
        "or shoulders, any belt or waist strap, and the boots with their own texture and lacing, "
        "all fitted the same closer-to-the-body way. The fit must stay tight enough that the "
        "shoulders, elbows, wrists, hips, knees, and ankles stay visually readable through the "
        "fabric, but every garment piece, color, material, and worn detail from the reference "
        "must still read clearly on the fitted body -- do not flatten the outfit into a single "
        "plain color or a smooth, unadorned surface. Do not invent new clothing, armor, or "
        "accessories, and do not change the garment style. Keep both legs straight and "
        "separated, with parallel feet facing forward. The weapon will be attached later in 3D."
    ),
    negative_prompt=(
        "cropped head, cropped feet, duplicate person, duplicate limbs, missing limbs, fused "
        "limbs, A-pose, bent arms, lowered arms, raised arms, crossed arms, arms touching torso, "
        "bent knees, crossed legs, seated pose, action pose, scenery occlusion, text, watermark, "
        "weapon, grip, sword, shield, object in either hand, closed hand, long dress, skirt, robe, "
        "cape, wide fabric crossing hips, fabric hiding knees, redesigned outfit, new clothing "
        "design, invented armor, ornate armor, fantasy armor, plate armor, different clothing "
        "colors, different clothing materials, losing wear and damage, plain bodysuit, blank "
        "bodysuit, featureless bodysuit, solid-color bodysuit, unadorned clothing, textureless "
        "clothing, smooth clothing with no detail, missing coat, missing hood, coat replaced "
        "with plain clothing, losing the coat, losing the hood, losing trim, losing seams, "
        "losing frayed edges, losing the collar, losing the harness, losing the belt, generic "
        "base layer, plain turtleneck, plain leggings, different skin tone, human skin, pink "
        "skin, pale skin, tan skin, changed eye color, blue eyes, different eye color, losing "
        "distinctive markings, losing cybernetics, losing circuitry, losing tattoos, generic "
        "person, different face shape, different individual"
    ),
    pose_guide_asset=t_pose_guide_asset(),
)

PROFILES = {
    HUMANOID_A_POSE_SWORD_V1.id: HUMANOID_A_POSE_SWORD_V1,
    HUMANOID_A_POSE_SWORD_V2.id: HUMANOID_A_POSE_SWORD_V2,
    HUMANOID_A_POSE_RIG_V3.id: HUMANOID_A_POSE_RIG_V3,
    HUMANOID_T_POSE_RIG_V1.id: HUMANOID_T_POSE_RIG_V1,
}

FLUX2_KLEIN_WORKFLOW_VERSION = "flux2-klein-image-edit-v1"
FLUX2_KLEIN_MULTI_REFERENCE_WORKFLOW_VERSION = (
    "flux2-klein-multi-reference-image-edit-v2"
)
FLUX2_KLEIN_MODEL = "flux-2-klein-base-4b-fp8.safetensors"


# View synthesis (workstream E3): given one approved front image, invent a
# genuine back/left/right view of the same character with the same
# image-conditioned Flux.2 Klein path the T-pose candidate uses. Per the
# project log, a mirrored/flipped copy of the front image makes downstream
# reconstruction WORSE than no side view at all -- these prompts explicitly
# reject a mirrored copy and demand a *redrawn*, genuinely different
# perspective that still preserves identity, outfit, and lighting.
#
# v2 (bead t0z17): v1 only said "standing in the same pose as the
# reference," which Flux.2 Klein interpreted loosely and relaxed a T-pose
# reference to arms-at-sides on every synthesized view -- the 4-view
# generate then logged an 18.3% bbox-height mismatch across views and the
# resulting mesh stood arms-down instead of T-posed. v2 makes the pose
# requirement literal and unambiguous (hold the reference's exact arm
# angle, elbow angle, leg stance, and hand position; a T-pose reference
# demands a T-pose result) and adds explicit pose-drift terms to the
# negative prompt.
# v3 (P1 image-adherence bug): v2's identity clause ("use the reference image
# only as the identity, hairstyle, and wardrobe reference") was not enough to
# stop the model from hallucinating a stock human when the requested view
# forces it to invent unseen geometry (the back of the head, an unseen
# profile). On an out-of-distribution source (a bald cyborg alien with a
# circuitry skull) the back-view redraw fell back to its training prior for
# "a person's back" -- a generic woman with long hair -- discarding the
# reference almost entirely, even though the same single-reference mechanism
# preserves identity fine under a pose-only change. v3 makes the anti-drift
# instruction explicit and literal: name the specific features that must
# survive (skin tone/markings, head/ear shape, hair length -- including
# "no hair" when the reference is bald) and forbid inventing a hairstyle or
# feature the reference does not show, instead of relying on a generic
# "same character" instruction the model can satisfy loosely.
# v4 (P1, LEFT/RIGHT collapse to the same ~3/4 angle): v3's left/right prompts
# named the side ("left-profile view", "the character's left side faces the
# camera") but gave no *geometric* anchor, and image models notoriously
# cannot resolve a character's left from their right by name alone --
# especially when the reference image is itself already a 3/4 turn, which
# gives the strong ReferenceLatent-conditioned identity channel a nearby
# viewpoint to collapse back onto. On the operator's alien source (zix.webp)
# both LEFT and RIGHT rendered as near-identical 3/4 turns facing the same
# way. v4 (left/right only; back is unaffected and stays v3) replaces the
# name-only instruction with a literal camera-placement anchor -- which edge
# of the frame the nose points toward, which ear faces the camera, which
# side must be entirely absent from the frame -- and adds "three-quarter
# view" (already in back's negative prompt but missing from left/right's) plus
# the opposite side's ear/eye/nose-direction terms to the negative prompt.
# Verified on the operator's zix.webp source across two seeds per side: nose
# direction and visible ear consistently swap between LEFT and RIGHT, and
# identity (skin tone, circuitry markings, coat) holds in every render.
# A follow-up pass verified the T-pose-hold requirement specifically (a
# strict side profile means one arm foreshortens toward the camera and the
# other extends away, which risks the far arm disappearing behind the
# torso): tested against a genuine T-pose reference (the humanoid-t-pose-
# rig-v1 output for the same source) across all three views. BACK and LEFT
# held the T-pose correctly on the first pass; the initial RIGHT render
# fully occluded the far arm behind the torso instead of dropping the pose,
# which the pose clause now names explicitly ("still visible past the
# silhouette of the torso, not hidden or cropped") with matching negative
# terms (arm hidden behind the torso, single arm visible, T-pose relaxed).
#
# v4 (P1, BACK invents long hair on a bald character): v3's "if the
# reference character is bald or hairless, the back of the head in this
# view must also be bald with no hair added" clause reads as a conditional
# aside buried inside a longer identity sentence -- easy for the model to
# drop when the requested view forces it to invent the one region (the
# crown and rear of the skull) the reference never shows at all. On the
# operator's live end-to-end synthesize-views job (`0197d1ff3de34831b4
# ef7f2d3f5adb1f`'s successor probe, source a bald cyborg alien with visible
# skull circuitry) the back view discarded the reference almost entirely
# and rendered a generic elf with long straight black hair and an invented
# ornate outfit -- the same training-prior fallback v3 was written to
# prevent, just not hardened enough to survive it. v4 makes bald-headedness
# a first-class, front-loaded, unconditional instruction for THIS
# character (matching the left/right v4 pattern of naming the literal
# anatomy instead of relying on a conditional identity clause the model can
# satisfy loosely) and doubles down on the circuitry markings as the
# concrete evidence of scalp coverage: "the back of the head in this view
# is a BALD head with bare scalp visible" plus "the circuitry panels seen
# on the front of the skull continue onto the rear of the scalp and down
# the neck" -- something a stock human/elf back-of-head render structurally
# cannot satisfy, forcing the model to keep rendering skin and circuitry
# rather than hair. Negative prompt gains explicit hair-shape terms (long
# hair, hairstyle, ponytail, braid, wig, hood up) beyond the generic
# "invented hairstyle" v3 already had, plus "different character" framed as
# its own standalone term rather than only inside a longer phrase.
VIEW_SYNTHESIS_BACK_V4 = ImagePreparationProfile(
    id="view-synthesis-back-v4",
    version=4,
    prompt=(
        "Use the reference image only as the identity and wardrobe reference for this "
        "character. Redraw this exact same individual seen from directly behind: a genuine "
        "rear view, not a mirrored or flipped copy of the reference image. This must be "
        "recognizably the same individual, not a different or generic person: keep the same "
        "skin tone and skin markings, the same head and ear shape, and the same distinctive "
        "features such as tattoos, cybernetics, or unusual skin texture. Head covering is the "
        "defining requirement of this view: if the reference character is bald or hairless, "
        "the back of the head in this view is a BALD head with bare scalp visible, no hair "
        "added, not covered by hair of any length -- if the reference shows circuitry, "
        "panels, tattoos, or other markings on the skull, those same markings continue onto "
        "the rear of the scalp and down the back of the neck, exactly as visible skin, not "
        "hidden under hair. Only "
        "render hair on this view if the reference image clearly shows hair; in that case "
        "match its exact length, color, and style. Keep the same outfit, same proportions, "
        "and same lighting on a plain neutral background. Show the full body from head to "
        "feet, holding the exact same body pose as the reference image: match the reference "
        "image's arm angle, elbow angle, leg stance, and hand position exactly -- if the "
        "reference has both arms held straight out horizontally in a T-pose, this back view "
        "must also show both arms held straight out horizontally at shoulder height, not "
        "lowered, not bent, and not resting against the torso. Only the camera angle changes. "
        "Face away from the camera so the back of the head, back of the clothing, and the "
        "back of the scalp (bald, with any markings continuing from the front) are visible."
    ),
    negative_prompt=(
        "front view, three-quarter view, face visible, mirrored copy of the reference "
        "image, flipped reference image, different character, different individual, generic "
        "person, unrelated person, stock model, different outfit, invented hairstyle, hair, "
        "long hair, hair added when the reference is bald, hairstyle, wig, ponytail, braid, "
        "bun, bangs, hood up, covered scalp, different hair length, different hair color, "
        "different skin tone, different face shape, losing distinctive markings, losing "
        "cybernetics, losing circuitry, circuitry not continuing onto the back of the head, "
        "losing tattoos, different proportions, changed pose, different pose, altered pose, "
        "arms lowered, arms at sides, arms down, relaxed arms, arms touching torso, bent "
        "elbows, straightened elbows, A-pose, crossed arms, hands on hips, different leg "
        "stance, cropped head, cropped feet, duplicate person, extra limbs, missing limbs, "
        "text, watermark, scenery, background objects"
    ),
    # v4 stress-test finding: against a well-hardened T-pose source (correct
    # bald+circuitry identity), v4's prompt hardening alone cleared the bar
    # cleanly across 3 seeds. But re-tested against the SAME weaker source
    # that produced the original rejected render (a T-pose candidate baked
    # before the identity/outfit fix landed -- bald crown, but a less
    # distinctly alien face), v4 still grew a ponytail/mane of hair from the
    # nape of the neck on all 3 seeds tried, even though the crown itself
    # stayed bald and circuitry-marked in every one -- a real, reproducible
    # gap, not a fluke. Escalated per instructions: the head/shoulders crop
    # is chained as a second ReferenceLatent (`head_crop_reference=True`)
    # to give the model an unmistakable close-up anchor for the exact
    # region it otherwise still has license to invent.
    head_crop_reference=True,
)

# v5 (P1, LEFT/RIGHT collapsed to the SAME frame direction on the operator's
# live final-gate set): the coordinator's own inspection found left.png and
# right.png both facing frame-right, with the source's asymmetric ear
# hardware/neck glow (visible on the character's own LEFT side in
# zix.webp) landing in the RIGHT-labeled output while LEFT showed a plainer
# ear. Root-causing this against the geometry v4 claimed to anchor:
#
#  1. v4's frame-edge polarity was backwards. Working the camera-placement
#     geometry through explicitly (character facing the front reference
#     camera; camera relocated to the character's own left flank for a
#     left-profile shot) puts the character's nose on the LEFT edge of the
#     resulting frame, not the right -- verified three independent ways
#     (vector cross-products in two different world-facing orientations,
#     plus a compass/photographer walk-through) and independently confirmed
#     against zix.webp itself: the raw source is already a near-profile
#     shot with the nose pointing to the frame's LEFT edge, and that same
#     shot is exactly where the ear hardware and neck glow are visible --
#     i.e. camera-at-left-flank produces nose-points-frame-left AND
#     hardware-visible simultaneously, which only holds if v4's "nose
#     points toward the RIGHT edge" for the LEFT profile (and the mirrored
#     claim for RIGHT) had the polarity swapped. v5 corrects it: LEFT's
#     nose points to the frame's LEFT edge, RIGHT's nose points to the
#     frame's RIGHT edge. (The ear/temple-faces-camera and opposite-side-
#     not-visible clauses were already correct in v4 and are unchanged.)
#  2. Polarity alone does not explain a same-direction COLLAPSE (a
#     consistent swap would still leave the two prompts producing opposite,
#     just mislabeled, directions) -- the live evidence is that Flux.2
#     Klein is not reliably tracking the "RIGHT edge"/"LEFT edge" word
#     swap at all, exactly the failure mode the v4 comment already
#     predicted ("image models notoriously cannot resolve a character's
#     left from their right by name alone"). Compounding this, v4's LEFT
#     and RIGHT prompts opened with an IDENTICAL ~140-character preamble
#     ("Use the reference image only as the identity and wardrobe
#     reference... Redraw this exact same individual in a strict
#     90-degree [x]-profile view...") before the direction-bearing clause
#     appeared, so the two conditionings started from a near-identical
#     embedding and diverged only on a same-position single-word swap deep
#     in the prompt. v5 front-loads the camera-placement/direction clause
#     as the FIRST sentence of each prompt so the two conditionings
#     diverge from the first token, and repeats the direction requirement
#     rather than relying on one polarity-sensitive word pair to do all the
#     work.
#  3. Direction alone is not the full failure: the asymmetric hardware
#     itself needs an explicit, generic (not character-specific) rule for
#     which profile shows it. v5 adds a conditional clause -- "if the
#     reference shows hardware on the character's LEFT side, THIS view
#     reproduces it / shows a plain side" -- to both prompts (mirrored),
#     matched with explicit negative-prompt terms for hardware appearing on
#     the wrong side or duplicated on both. This is deliberately phrased as
#     a conditional so the same profile text works for any source, not
#     just this alien reference.
#  4. Escalated per the back-view v4 precedent (bead ncxzd's mechanism):
#     head_crop_reference=True is now set on LEFT and RIGHT too, chaining
#     the reference's own head/shoulders crop as a second, symmetric
#     ReferenceLatent pair so the model has a close-up, hard-to-ignore
#     anchor for exactly the ear/neck region whose asymmetry it otherwise
#     has to infer from the full-body reference alone.
#
# v6 (measured against zix.webp, 12 seeds -- 6 LEFT, 6 RIGHT, all against the
# same T-pose candidate): v5 fixed the direction collapse completely (12/12
# renders pointed the correct way) and the HARDWARE-side got it right most
# of the time (5/6 LEFT renders showed the device; 1/6 omitted it), but the
# PLAIN side leaked invented hardware at a high rate -- 4/6 RIGHT renders
# added some cyborg device (a neck collar, a wired implant, an in-ear
# cable) even though the plain side must show none of it. The v5 clause
# scoped the "no hardware" instruction to "a plain, unmodified ear and
# neck," which reads as ruling out an ear-specific device but leaves room
# for a collar or implant elsewhere on the same side's neck/throat -- and
# that is exactly where every v5 leak appeared. v6 widens the "plain side"
# clause to an exhaustive, explicit list (device, implant, cable, wire,
# connector, collar, glowing marking) covering the whole head AND neck, not
# just the ear, and adds matching negative-prompt terms for a neck collar,
# an implant, and an in-ear cable specifically (the concrete shapes the
# leaks took). It also drops the earlier "matters and is not optional"
# hedge language for a flatter imperative on both branches, since the
# leak was asymmetric: the SHOW branch was already followed most of the
# time; the SUPPRESS branch needed the stronger wording.
#
# v7 (2026-09-02, bead idn1d): two consecutive T-pose sets rendered both
# profiles as backlit near-black silhouettes (subject luminance 8-29 vs 57-63
# on the front), and the shape-only consistency check scored them clean. The
# trigger was our own wording: v6 said "genuine profile SILHOUETTE" -- an
# instruction the model eventually took literally -- and the lone lighting
# clause ("same lighting on a plain neutral background") was buried and weak.
# v7 purges the word silhouette from both profiles (profile "view"; torso
# "outline"), replaces the buried clause with an explicit fully-lit anchor,
# and adds silhouette/backlit/underexposed negative terms. Paired with the
# new value_collapse check in view_consistency (calibrated on the two failed
# sets) so a bad roll is caught at the tile instead of shipping.
VIEW_SYNTHESIS_LEFT_V7 = ImagePreparationProfile(
    id="view-synthesis-left-v7",
    version=7,
    prompt=(
        "Camera placement is the defining requirement of this view: position the camera "
        "directly to the character's own left side, in a strict 90-degree left-profile "
        "shot, so the character's face and nose point toward the LEFT edge of the frame, "
        "the left ear and left temple face squarely toward the camera, and no part of the "
        "right side of the face, the right ear, or the right eye is visible. This is a "
        "genuine profile view seen from the character's own left side, not a "
        "three-quarter angle, not a frontal view, and not a mirrored or flipped copy of "
        "the reference image -- the nose points LEFT in this image, never right. "
        "Use the reference image only as the identity and wardrobe reference for this "
        "character; redraw this exact same individual in this new camera position. This "
        "must be recognizably the same individual, not a different or generic person: "
        "keep the same skin tone and skin markings, the same head and ear shape, the same "
        "distinctive features such as tattoos, cybernetics, or unusual skin texture, and "
        "the exact same hair length and color as the reference -- if the reference "
        "character is bald or hairless, this view must also be bald with no hair added. "
        "Check the reference image and the close-up head reference for any attached "
        "device, implant, cable, wire, connector, collar, or glowing marking on the head "
        "or neck. If the reference character has hardware like this and it sits on the "
        "character's LEFT side, this view MUST show it: render that exact device, its "
        "cable routing, and its glow, clearly and fully visible on the left ear and down "
        "the left side of the neck, matching the reference precisely. If instead that "
        "same hardware sits on the character's RIGHT side in the reference, this "
        "left-profile view shows the OPPOSITE, plain side: render a completely plain, "
        "unmodified left ear and a bare left side of the neck -- no device, no implant, "
        "no cable, no wire, no connector, no collar, and no glowing marking anywhere on "
        "this view's head or neck, because all of that belongs only to the other side of "
        "the head, which this view does not show. Keep the same outfit and the same "
        "proportions. Light this view exactly like the reference: bright, even, "
        "neutral studio lighting with the subject FULLY LIT and clearly visible -- "
        "every garment color, every skin detail, and every marking readable at full "
        "brightness. Never render the subject backlit, in shadow, or as a dark or "
        "silhouetted figure; the background stays plain light neutral and the subject "
        "stays lit. Show the full body from head to feet, holding the exact same body "
        "pose as the reference image: match the reference image's arm angle, elbow "
        "angle, leg stance, and hand position exactly -- if the reference has both arms "
        "held straight out horizontally in a T-pose, this left-profile view must also "
        "hold that exact T-pose: both arms straight out horizontally at shoulder height, "
        "one arm extending toward the camera and foreshortened toward the viewer, the "
        "other arm extending directly away from the camera and still visible past the "
        "outline of the torso, not hidden or cropped by the body -- never lowered, "
        "never bent, never brought in to the torso, and the T-pose never relaxed. Only "
        "the camera angle changes."
    ),
    negative_prompt=(
        "silhouette, silhouetted figure, backlit, backlighting, rim lighting, rim-lit "
        "figure, underexposed, dark figure, shadow figure, black figure, subject in "
        "shadow, low-key lighting, dramatic lighting, "
        "front view, three-quarter view, quarter view, frontal view, facing the camera, "
        "angled toward the camera, same viewpoint as the reference image, back view, "
        "right-profile view, right ear visible, right side of face visible, right eye "
        "visible, nose pointing toward the RIGHT edge of the frame, nose facing right, "
        "mirrored copy of the reference image, flipped reference image, different "
        "character, different individual, generic person, unrelated person, stock model, "
        "different outfit, invented hairstyle, hair added when the reference is bald, "
        "wig, different hair length, different hair color, different skin tone, "
        "different face shape, losing distinctive markings, losing cybernetics, losing "
        "tattoos, different proportions, changed pose, different pose, altered pose, arms "
        "lowered, arms at sides, arms down, relaxed arms, arms touching torso, bent "
        "elbows, straightened elbows, A-pose, crossed arms, hands on hips, different leg "
        "stance, arm hidden behind the torso, missing arm, single arm visible, only one "
        "arm visible, T-pose relaxed, T-pose abandoned, arm brought forward, arm crossed "
        "in front of the body, hardware on the wrong side, hardware on the right ear "
        "instead of the left, hardware duplicated on both ears, hardware duplicated on "
        "both sides of the neck, missing hardware when the reference shows it on the "
        "left, hardware invented on the left when the reference shows it on the right, "
        "symmetric hardware on both sides, neck collar when the hardware belongs on the "
        "other side, implant collar, cyborg collar, glowing neck implant, wire around the "
        "neck, cable around the neck, cable connector, attached electronics, in-ear "
        "device, earbud, ear cable, invented cyborg hardware not present in the "
        "reference, cropped head, cropped feet, duplicate person, extra limbs, missing "
        "limbs, text, watermark, scenery, background objects"
    ),
    head_crop_reference=True,
)

VIEW_SYNTHESIS_RIGHT_V7 = ImagePreparationProfile(
    id="view-synthesis-right-v7",
    version=7,
    prompt=(
        "Camera placement is the defining requirement of this view: position the camera "
        "directly to the character's own right side, in a strict 90-degree right-profile "
        "shot, so the character's face and nose point toward the RIGHT edge of the "
        "frame, the right ear and right temple face squarely toward the camera, and no "
        "part of the left side of the face, the left ear, or the left eye is visible. "
        "This is a genuine profile view seen from the character's own right side, "
        "not a three-quarter angle, not a frontal view, and not a mirrored or flipped "
        "copy of the reference image -- the nose points RIGHT in this image, never left. "
        "Use the reference image only as the identity and wardrobe reference for this "
        "character; redraw this exact same individual in this new camera position. This "
        "must be recognizably the same individual, not a different or generic person: "
        "keep the same skin tone and skin markings, the same head and ear shape, the "
        "same distinctive features such as tattoos, cybernetics, or unusual skin "
        "texture, and the exact same hair length and color as the reference -- if the "
        "reference character is bald or hairless, this view must also be bald with no "
        "hair added. Check the reference image and the close-up head reference for any "
        "attached device, implant, cable, wire, connector, collar, or glowing marking on "
        "the head or neck. If the reference character has hardware like this and it sits "
        "on the character's RIGHT side, this view MUST show it: render that exact "
        "device, its cable routing, and its glow, clearly and fully visible on the right "
        "ear and down the right side of the neck, matching the reference precisely. If "
        "instead that same hardware sits on the character's LEFT side in the reference, "
        "this right-profile view shows the OPPOSITE, plain side: render a completely "
        "plain, unmodified right ear and a bare right side of the neck -- no device, no "
        "implant, no cable, no wire, no connector, no collar, and no glowing marking "
        "anywhere on this view's head or neck, because all of that belongs only to the "
        "other side of the head, which this view does not show. Keep the same outfit "
        "and the same proportions. Light this view exactly like the reference: bright, "
        "even, neutral studio lighting with the subject FULLY LIT and clearly visible "
        "-- every garment color, every skin detail, and every marking readable at full "
        "brightness. Never render the subject backlit, in shadow, or as a dark or "
        "silhouetted figure; the background stays plain light neutral and the subject "
        "stays lit. Show the full body from head to "
        "feet, holding the exact same body pose as the reference image: match the "
        "reference image's arm angle, elbow angle, leg stance, and hand position "
        "exactly -- if the reference has both arms held straight out horizontally in a "
        "T-pose, this right-profile view must also hold that exact T-pose: both arms "
        "straight out horizontally at shoulder height, one arm extending toward the "
        "camera and foreshortened toward the viewer, the other arm extending directly "
        "away from the camera and still visible past the outline of the torso, not "
        "hidden or cropped by the body -- never lowered, never bent, never brought in to "
        "the torso, and the T-pose never relaxed. Only the camera angle changes."
    ),
    negative_prompt=(
        "silhouette, silhouetted figure, backlit, backlighting, rim lighting, rim-lit "
        "figure, underexposed, dark figure, shadow figure, black figure, subject in "
        "shadow, low-key lighting, dramatic lighting, "
        "front view, three-quarter view, quarter view, frontal view, facing the camera, "
        "angled toward the camera, same viewpoint as the reference image, back view, "
        "left-profile view, left ear visible, left side of face visible, left eye "
        "visible, nose pointing toward the LEFT edge of the frame, nose facing left, "
        "mirrored copy of the reference image, flipped reference image, different "
        "character, different individual, generic person, unrelated person, stock model, "
        "different outfit, invented hairstyle, hair added when the reference is bald, "
        "wig, different hair length, different hair color, different skin tone, "
        "different face shape, losing distinctive markings, losing cybernetics, losing "
        "tattoos, different proportions, changed pose, different pose, altered pose, arms "
        "lowered, arms at sides, arms down, relaxed arms, arms touching torso, bent "
        "elbows, straightened elbows, A-pose, crossed arms, hands on hips, different leg "
        "stance, arm hidden behind the torso, missing arm, single arm visible, only one "
        "arm visible, T-pose relaxed, T-pose abandoned, arm brought forward, arm crossed "
        "in front of the body, hardware on the wrong side, hardware on the left ear "
        "instead of the right, hardware duplicated on both ears, hardware duplicated on "
        "both sides of the neck, missing hardware when the reference shows it on the "
        "right, hardware invented on the right when the reference shows it on the left, "
        "symmetric hardware on both sides, neck collar when the hardware belongs on the "
        "other side, implant collar, cyborg collar, glowing neck implant, wire around the "
        "neck, cable around the neck, cable connector, attached electronics, in-ear "
        "device, earbud, ear cable, invented cyborg hardware not present in the "
        "reference, cropped head, cropped feet, duplicate person, extra limbs, missing "
        "limbs, text, watermark, scenery, background objects"
    ),
    head_crop_reference=True,
)

VIEW_SYNTHESIS_PROFILES: dict[str, ImagePreparationProfile] = {
    "back": VIEW_SYNTHESIS_BACK_V4,
    "left": VIEW_SYNTHESIS_LEFT_V7,
    "right": VIEW_SYNTHESIS_RIGHT_V7,
}

# Source-pose family: the "tpose" profiles above are hardened for a T-pose
# front and actively re-impose it ("if the reference has both arms held
# straight out horizontally in a T-pose, this view must also...", with a
# matching negative-prompt T-pose enforcement list) -- run against a raw
# ¾ photo with arms down and a held weapon, they redraw an A/T-pose and drop
# the weapon (2026-09-05 live run). This family exists for the identity-first
# reconstruction path: it redraws the character in the EXACT source pose
# (arms as photographed, held objects kept) from behind/left/right. The
# tpose family must never be used against a non-T-pose reference, and this
# family must never be used against a genuine T-pose front.
SOURCE_POSE_VIEW_BACK_V1 = ImagePreparationProfile(
    id="source-pose-view-back-v1",
    version=1,
    prompt=(
        "Use the reference image only as the identity and wardrobe reference for this "
        "character. Redraw this exact same individual seen from directly behind: a genuine "
        "rear view, not a mirrored or flipped copy of the reference image. This must be "
        "recognizably the same individual, not a different or generic person: keep the same "
        "skin tone and skin markings, the same head and ear shape, and the same distinctive "
        "features such as tattoos, cybernetics, or unusual skin texture. Head covering is the "
        "defining requirement of this view: if the reference character is bald or hairless, "
        "the back of the head in this view is a BALD head with bare scalp visible, no hair "
        "added, not covered by hair of any length -- if the reference shows circuitry, "
        "panels, tattoos, or other markings on the skull, those same markings continue onto "
        "the rear of the scalp and down the back of the neck, exactly as visible skin, not "
        "hidden under hair. Only "
        "render hair on this view if the reference image clearly shows hair; in that case "
        "match its exact length, color, and style. Keep the same outfit, same proportions, "
        "and same lighting on a plain neutral background. Show the full body from head to "
        "feet, holding the exact same body pose as the reference image: the same arm "
        "positions, elbow angles, hand positions, leg stance and head tilt -- if the "
        "reference's arms hang at the sides, they hang at the sides here; if the reference "
        "holds an object (a weapon, a tool, a staff), the same object is held in the same "
        "hand at the same angle, seen from this camera. Do not straighten, raise, or spread "
        "the arms. Only the camera angle changes. "
        "Face away from the camera so the back of the head, back of the clothing, and the "
        "back of the scalp (bald, with any markings continuing from the front) are visible."
    ),
    negative_prompt=(
        "front view, three-quarter view, face visible, mirrored copy of the reference "
        "image, flipped reference image, different character, different individual, generic "
        "person, unrelated person, stock model, different outfit, invented hairstyle, hair, "
        "long hair, hair added when the reference is bald, hairstyle, wig, ponytail, braid, "
        "bun, bangs, hood up, covered scalp, different hair length, different hair color, "
        "different skin tone, different face shape, losing distinctive markings, losing "
        "cybernetics, losing circuitry, circuitry not continuing onto the back of the head, "
        "losing tattoos, different proportions, T-pose, A-pose, arms raised, arms spread, "
        "arms held out horizontally, outstretched arms, changed arm position, changed pose, "
        "different pose, altered pose, held object missing, weapon missing, empty hands when "
        "the reference holds something, hands on hips, crossed arms, different leg stance, "
        "cropped head, cropped feet, duplicate person, extra limbs, missing limbs, "
        "text, watermark, scenery, background objects"
    ),
    head_crop_reference=True,
)

SOURCE_POSE_VIEW_LEFT_V1 = ImagePreparationProfile(
    id="source-pose-view-left-v1",
    version=1,
    prompt=(
        "Camera placement is the defining requirement of this view: position the camera "
        "directly to the character's own left side, in a strict 90-degree left-profile "
        "shot, so the character's face and nose point toward the LEFT edge of the frame, "
        "the left ear and left temple face squarely toward the camera, and no part of the "
        "right side of the face, the right ear, or the right eye is visible. This is a "
        "genuine profile view seen from the character's own left side, not a "
        "three-quarter angle, not a frontal view, and not a mirrored or flipped copy of "
        "the reference image -- the nose points LEFT in this image, never right. "
        "Use the reference image only as the identity and wardrobe reference for this "
        "character; redraw this exact same individual in this new camera position. This "
        "must be recognizably the same individual, not a different or generic person: "
        "keep the same skin tone and skin markings, the same head and ear shape, the same "
        "distinctive features such as tattoos, cybernetics, or unusual skin texture, and "
        "the exact same hair length and color as the reference -- if the reference "
        "character is bald or hairless, this view must also be bald with no hair added. "
        "Check the reference image and the close-up head reference for any attached "
        "device, implant, cable, wire, connector, collar, or glowing marking on the head "
        "or neck. If the reference character has hardware like this and it sits on the "
        "character's LEFT side, this view MUST show it: render that exact device, its "
        "cable routing, and its glow, clearly and fully visible on the left ear and down "
        "the left side of the neck, matching the reference precisely. If instead that "
        "same hardware sits on the character's RIGHT side in the reference, this "
        "left-profile view shows the OPPOSITE, plain side: render a completely plain, "
        "unmodified left ear and a bare left side of the neck -- no device, no implant, "
        "no cable, no wire, no connector, no collar, and no glowing marking anywhere on "
        "this view's head or neck, because all of that belongs only to the other side of "
        "the head, which this view does not show. Keep the same outfit and the same "
        "proportions. Light this view exactly like the reference: bright, even, "
        "neutral studio lighting with the subject FULLY LIT and clearly visible -- "
        "every garment color, every skin detail, and every marking readable at full "
        "brightness. Never render the subject backlit, in shadow, or as a dark or "
        "silhouetted figure; the background stays plain light neutral and the subject "
        "stays lit. Show the full body from head to feet, holding the exact same body "
        "pose as the reference image: the same arm positions, elbow angles, hand "
        "positions, leg stance and head tilt -- if the reference's arms hang at the "
        "sides, they hang at the sides here; if the reference holds an object (a weapon, "
        "a tool, a staff), the same object is held in the same hand at the same angle, "
        "seen from this camera. Do not straighten, raise, or spread the arms. Only "
        "the camera angle changes."
    ),
    negative_prompt=(
        "silhouette, silhouetted figure, backlit, backlighting, rim lighting, rim-lit "
        "figure, underexposed, dark figure, shadow figure, black figure, subject in "
        "shadow, low-key lighting, dramatic lighting, "
        "front view, three-quarter view, quarter view, frontal view, facing the camera, "
        "angled toward the camera, same viewpoint as the reference image, back view, "
        "right-profile view, right ear visible, right side of face visible, right eye "
        "visible, nose pointing toward the RIGHT edge of the frame, nose facing right, "
        "mirrored copy of the reference image, flipped reference image, different "
        "character, different individual, generic person, unrelated person, stock model, "
        "different outfit, invented hairstyle, hair added when the reference is bald, "
        "wig, different hair length, different hair color, different skin tone, "
        "different face shape, losing distinctive markings, losing cybernetics, losing "
        "tattoos, different proportions, T-pose, A-pose, arms raised, arms spread, "
        "arms held out horizontally, outstretched arms, changed arm position, changed "
        "pose, different pose, altered pose, held object missing, weapon missing, empty "
        "hands when the reference holds something, hands on hips, crossed arms, "
        "different leg stance, hardware on the wrong side, hardware on the right ear "
        "instead of the left, hardware duplicated on both ears, hardware duplicated on "
        "both sides of the neck, missing hardware when the reference shows it on the "
        "left, hardware invented on the left when the reference shows it on the right, "
        "symmetric hardware on both sides, neck collar when the hardware belongs on the "
        "other side, implant collar, cyborg collar, glowing neck implant, wire around the "
        "neck, cable around the neck, cable connector, attached electronics, in-ear "
        "device, earbud, ear cable, invented cyborg hardware not present in the "
        "reference, cropped head, cropped feet, duplicate person, extra limbs, missing "
        "limbs, text, watermark, scenery, background objects"
    ),
    head_crop_reference=True,
)

SOURCE_POSE_VIEW_RIGHT_V1 = ImagePreparationProfile(
    id="source-pose-view-right-v1",
    version=1,
    prompt=(
        "Camera placement is the defining requirement of this view: position the camera "
        "directly to the character's own right side, in a strict 90-degree right-profile "
        "shot, so the character's face and nose point toward the RIGHT edge of the "
        "frame, the right ear and right temple face squarely toward the camera, and no "
        "part of the left side of the face, the left ear, or the left eye is visible. "
        "This is a genuine profile view seen from the character's own right side, "
        "not a three-quarter angle, not a frontal view, and not a mirrored or flipped "
        "copy of the reference image -- the nose points RIGHT in this image, never left. "
        "Use the reference image only as the identity and wardrobe reference for this "
        "character; redraw this exact same individual in this new camera position. This "
        "must be recognizably the same individual, not a different or generic person: "
        "keep the same skin tone and skin markings, the same head and ear shape, the "
        "same distinctive features such as tattoos, cybernetics, or unusual skin "
        "texture, and the exact same hair length and color as the reference -- if the "
        "reference character is bald or hairless, this view must also be bald with no "
        "hair added. Check the reference image and the close-up head reference for any "
        "attached device, implant, cable, wire, connector, collar, or glowing marking on "
        "the head or neck. If the reference character has hardware like this and it sits "
        "on the character's RIGHT side, this view MUST show it: render that exact "
        "device, its cable routing, and its glow, clearly and fully visible on the right "
        "ear and down the right side of the neck, matching the reference precisely. If "
        "instead that same hardware sits on the character's LEFT side in the reference, "
        "this right-profile view shows the OPPOSITE, plain side: render a completely "
        "plain, unmodified right ear and a bare right side of the neck -- no device, no "
        "implant, no cable, no wire, no connector, no collar, and no glowing marking "
        "anywhere on this view's head or neck, because all of that belongs only to the "
        "other side of the head, which this view does not show. Keep the same outfit "
        "and the same proportions. Light this view exactly like the reference: bright, "
        "even, neutral studio lighting with the subject FULLY LIT and clearly visible "
        "-- every garment color, every skin detail, and every marking readable at full "
        "brightness. Never render the subject backlit, in shadow, or as a dark or "
        "silhouetted figure; the background stays plain light neutral and the subject "
        "stays lit. Show the full body from head to "
        "feet, holding the exact same body pose as the reference image: the same arm "
        "positions, elbow angles, hand positions, leg stance and head tilt -- if the "
        "reference's arms hang at the sides, they hang at the sides here; if the "
        "reference holds an object (a weapon, a tool, a staff), the same object is held "
        "in the same hand at the same angle, seen from this camera. Do not straighten, "
        "raise, or spread the arms. Only the camera angle changes."
    ),
    negative_prompt=(
        "silhouette, silhouetted figure, backlit, backlighting, rim lighting, rim-lit "
        "figure, underexposed, dark figure, shadow figure, black figure, subject in "
        "shadow, low-key lighting, dramatic lighting, "
        "front view, three-quarter view, quarter view, frontal view, facing the camera, "
        "angled toward the camera, same viewpoint as the reference image, back view, "
        "left-profile view, left ear visible, left side of face visible, left eye "
        "visible, nose pointing toward the LEFT edge of the frame, nose facing left, "
        "mirrored copy of the reference image, flipped reference image, different "
        "character, different individual, generic person, unrelated person, stock model, "
        "different outfit, invented hairstyle, hair added when the reference is bald, "
        "wig, different hair length, different hair color, different skin tone, "
        "different face shape, losing distinctive markings, losing cybernetics, losing "
        "tattoos, different proportions, T-pose, A-pose, arms raised, arms spread, "
        "arms held out horizontally, outstretched arms, changed arm position, changed "
        "pose, different pose, altered pose, held object missing, weapon missing, empty "
        "hands when the reference holds something, hands on hips, crossed arms, "
        "different leg stance, hardware on the wrong side, hardware on the left ear "
        "instead of the right, hardware duplicated on both ears, hardware duplicated on "
        "both sides of the neck, missing hardware when the reference shows it on the "
        "right, hardware invented on the right when the reference shows it on the left, "
        "symmetric hardware on both sides, neck collar when the hardware belongs on the "
        "other side, implant collar, cyborg collar, glowing neck implant, wire around the "
        "neck, cable around the neck, cable connector, attached electronics, in-ear "
        "device, earbud, ear cable, invented cyborg hardware not present in the "
        "reference, cropped head, cropped feet, duplicate person, extra limbs, missing "
        "limbs, text, watermark, scenery, background objects"
    ),
    head_crop_reference=True,
)

SOURCE_POSE_VIEW_SYNTHESIS_PROFILES: dict[str, ImagePreparationProfile] = {
    "back": SOURCE_POSE_VIEW_BACK_V1,
    "left": SOURCE_POSE_VIEW_LEFT_V1,
    "right": SOURCE_POSE_VIEW_RIGHT_V1,
}

# Mesh-guided family: independent per-view redraws (the tpose/source-pose
# families above) cannot be registered to a mesh -- Hunyuan mv measured
# left/right IoU 0.43/0.62 across them. Generating each view by PAINTING a
# clay render of the mesh from that exact camera makes the result consistent
# with the geometry by construction; identity comes from a separate photo
# reference chained via `identity_reference=True`, not from the (headless,
# untextured) clay render itself.
_MESH_GUIDED_PROMPT_BODY = (
    "Paint this exact clay render as the character in the reference photo. Keep the clay "
    "render's silhouette, pose, proportions, limb positions and camera angle pixel for "
    "pixel: do not move, resize, rotate or redraw any body part, and do not add or remove "
    "anything from the silhouette. Take everything about appearance -- skin tone and "
    "markings, face, ears, cybernetics, hair or bald scalp, clothing colours, materials, "
    "wear and details -- from the reference photo of the same character. Photoreal, plain "
    "neutral background, even studio lighting, no cast shadows on the background."
)

# Reused verbatim from VIEW_SYNTHESIS_BACK_V4: the model's training prior for
# "a person's back" fills in invented hair over a bald/circuitry-marked skull
# unless the requirement is named as a first-class, unconditional sentence.
_MESH_GUIDED_BALD_SCALP_SENTENCES = (
    "Head covering is the defining requirement of this view: if the reference character is "
    "bald or hairless, the back of the head in this view is a BALD head with bare scalp "
    "visible, no hair added, not covered by hair of any length -- if the reference shows "
    "circuitry, panels, tattoos, or other markings on the skull, those same markings "
    "continue onto the rear of the scalp and down the back of the neck, exactly as visible "
    "skin, not hidden under hair. Only render hair on this view if the reference image "
    "clearly shows hair; in that case match its exact length, color, and style."
)

_MESH_GUIDED_NEGATIVE_PROMPT = (
    "changed silhouette, changed pose, moved limbs, different proportions, different "
    "camera angle, redrawn body, extra limbs, missing limbs, grey clay left unpainted, "
    "untextured, different character, different outfit, hair added when the reference is "
    "bald, text, watermark, background objects, cast shadows"
)

# Indexed prompting (identity-variants lever 1): Flux.2 Klein receives the
# clay render and the identity photo (and, when chained, the head crop) as
# separate LoadImage nodes with no inherent labels -- naming them by
# position at the very start of the prompt removes the ambiguity of which
# image is the structural reference and which is the identity to paint.
_MESH_GUIDED_INDEXED_INTRO = (
    "The first image is an untextured grey clay render of a 3D model. The second image "
    "is the reference photo of the character to paint. "
)
_MESH_GUIDED_INDEXED_INTRO_WITH_HEAD_CROP = (
    _MESH_GUIDED_INDEXED_INTRO
    + "The third image is a close-up of the same character's head. "
)

_MESH_GUIDED_FRONT_SENTENCE = "This is the character's own front view, facing the camera directly. "
_MESH_GUIDED_BACK_SENTENCE = "This is the character's own back view, seen from directly behind. "
_MESH_GUIDED_LEFT_SENTENCE = "This is the character's own left side in strict profile. "
_MESH_GUIDED_RIGHT_SENTENCE = "This is the character's own right side in strict profile. "
_MESH_GUIDED_FRONT_LEFT_SENTENCE = (
    "This is the character seen from the front-left, a three-quarter view. "
)
_MESH_GUIDED_FRONT_RIGHT_SENTENCE = (
    "This is the character seen from the front-right, a three-quarter view. "
)
_MESH_GUIDED_BACK_LEFT_SENTENCE = (
    "This is the character seen from the back-left, a three-quarter rear view. "
)
_MESH_GUIDED_BACK_RIGHT_SENTENCE = (
    "This is the character seen from the back-right, a three-quarter rear view. "
)

MESH_GUIDED_VIEW_FRONT_V1 = ImagePreparationProfile(
    id="mesh-guided-view-front-v1",
    version=1,
    prompt=(
        _MESH_GUIDED_INDEXED_INTRO_WITH_HEAD_CROP
        + _MESH_GUIDED_FRONT_SENTENCE
        + _MESH_GUIDED_PROMPT_BODY
    ),
    negative_prompt=_MESH_GUIDED_NEGATIVE_PROMPT,
    identity_reference=True,
    head_crop_reference=True,
)

MESH_GUIDED_VIEW_BACK_V1 = ImagePreparationProfile(
    id="mesh-guided-view-back-v1",
    version=1,
    prompt=(
        _MESH_GUIDED_INDEXED_INTRO
        + _MESH_GUIDED_BACK_SENTENCE
        + _MESH_GUIDED_PROMPT_BODY
        + " "
        + _MESH_GUIDED_BALD_SCALP_SENTENCES
    ),
    negative_prompt=_MESH_GUIDED_NEGATIVE_PROMPT,
    identity_reference=True,
)

MESH_GUIDED_VIEW_LEFT_V1 = ImagePreparationProfile(
    id="mesh-guided-view-left-v1",
    version=1,
    prompt=(
        _MESH_GUIDED_INDEXED_INTRO_WITH_HEAD_CROP
        + _MESH_GUIDED_LEFT_SENTENCE
        + _MESH_GUIDED_PROMPT_BODY
    ),
    negative_prompt=_MESH_GUIDED_NEGATIVE_PROMPT,
    identity_reference=True,
    head_crop_reference=True,
)

MESH_GUIDED_VIEW_RIGHT_V1 = ImagePreparationProfile(
    id="mesh-guided-view-right-v1",
    version=1,
    prompt=(
        _MESH_GUIDED_INDEXED_INTRO_WITH_HEAD_CROP
        + _MESH_GUIDED_RIGHT_SENTENCE
        + _MESH_GUIDED_PROMPT_BODY
    ),
    negative_prompt=_MESH_GUIDED_NEGATIVE_PROMPT,
    identity_reference=True,
    head_crop_reference=True,
)

MESH_GUIDED_VIEW_FRONT_LEFT_V1 = ImagePreparationProfile(
    id="mesh-guided-view-front-left-v1",
    version=1,
    prompt=(
        _MESH_GUIDED_INDEXED_INTRO_WITH_HEAD_CROP
        + _MESH_GUIDED_FRONT_LEFT_SENTENCE
        + _MESH_GUIDED_PROMPT_BODY
    ),
    negative_prompt=_MESH_GUIDED_NEGATIVE_PROMPT,
    identity_reference=True,
    head_crop_reference=True,
)

MESH_GUIDED_VIEW_FRONT_RIGHT_V1 = ImagePreparationProfile(
    id="mesh-guided-view-front-right-v1",
    version=1,
    prompt=(
        _MESH_GUIDED_INDEXED_INTRO_WITH_HEAD_CROP
        + _MESH_GUIDED_FRONT_RIGHT_SENTENCE
        + _MESH_GUIDED_PROMPT_BODY
    ),
    negative_prompt=_MESH_GUIDED_NEGATIVE_PROMPT,
    identity_reference=True,
    head_crop_reference=True,
)

MESH_GUIDED_VIEW_BACK_LEFT_V1 = ImagePreparationProfile(
    id="mesh-guided-view-back-left-v1",
    version=1,
    prompt=(
        _MESH_GUIDED_INDEXED_INTRO
        + _MESH_GUIDED_BACK_LEFT_SENTENCE
        + _MESH_GUIDED_PROMPT_BODY
        + " "
        + _MESH_GUIDED_BALD_SCALP_SENTENCES
    ),
    negative_prompt=_MESH_GUIDED_NEGATIVE_PROMPT,
    identity_reference=True,
)

MESH_GUIDED_VIEW_BACK_RIGHT_V1 = ImagePreparationProfile(
    id="mesh-guided-view-back-right-v1",
    version=1,
    prompt=(
        _MESH_GUIDED_INDEXED_INTRO
        + _MESH_GUIDED_BACK_RIGHT_SENTENCE
        + _MESH_GUIDED_PROMPT_BODY
        + " "
        + _MESH_GUIDED_BALD_SCALP_SENTENCES
    ),
    negative_prompt=_MESH_GUIDED_NEGATIVE_PROMPT,
    identity_reference=True,
)

MESH_GUIDED_VIEW_SYNTHESIS_PROFILES: dict[str, ImagePreparationProfile] = {
    "front": MESH_GUIDED_VIEW_FRONT_V1,
    "back": MESH_GUIDED_VIEW_BACK_V1,
    "left": MESH_GUIDED_VIEW_LEFT_V1,
    "right": MESH_GUIDED_VIEW_RIGHT_V1,
    "front-left": MESH_GUIDED_VIEW_FRONT_LEFT_V1,
    "front-right": MESH_GUIDED_VIEW_FRONT_RIGHT_V1,
    "back-left": MESH_GUIDED_VIEW_BACK_LEFT_V1,
    "back-right": MESH_GUIDED_VIEW_BACK_RIGHT_V1,
}

# Identity-first family (lever 3): the same eight views, but the character
# photo is named as the FIRST image and chains onto the base conditioning
# before the clay render (ImagePreparationProfile.identity_first=True) --
# see build_flux2_klein_image_edit's identity_first branch. This makes the
# photo CFG's primary edit image instead of a secondary reference chained
# after the (headless, untextured) clay latent.
_MESH_GUIDED_IDFIRST_INTRO = (
    "The first image is the reference photo of the character. The second image is an "
    "untextured grey clay render of a 3D model of the same character from a new camera: "
    "paint the character onto that clay render, keeping the clay render's silhouette, "
    "pose, proportions and camera pixel for pixel. "
)

MESH_GUIDED_IDFIRST_VIEW_FRONT_V1 = ImagePreparationProfile(
    id="mesh-guided-idfirst-view-front-v1",
    version=1,
    prompt=(
        _MESH_GUIDED_IDFIRST_INTRO
        + _MESH_GUIDED_FRONT_SENTENCE
        + _MESH_GUIDED_PROMPT_BODY
    ),
    negative_prompt=_MESH_GUIDED_NEGATIVE_PROMPT,
    identity_reference=True,
    head_crop_reference=True,
    identity_first=True,
)

MESH_GUIDED_IDFIRST_VIEW_BACK_V1 = ImagePreparationProfile(
    id="mesh-guided-idfirst-view-back-v1",
    version=1,
    prompt=(
        _MESH_GUIDED_IDFIRST_INTRO
        + _MESH_GUIDED_BACK_SENTENCE
        + _MESH_GUIDED_PROMPT_BODY
        + " "
        + _MESH_GUIDED_BALD_SCALP_SENTENCES
    ),
    negative_prompt=_MESH_GUIDED_NEGATIVE_PROMPT,
    identity_reference=True,
    identity_first=True,
)

MESH_GUIDED_IDFIRST_VIEW_LEFT_V1 = ImagePreparationProfile(
    id="mesh-guided-idfirst-view-left-v1",
    version=1,
    prompt=(
        _MESH_GUIDED_IDFIRST_INTRO
        + _MESH_GUIDED_LEFT_SENTENCE
        + _MESH_GUIDED_PROMPT_BODY
    ),
    negative_prompt=_MESH_GUIDED_NEGATIVE_PROMPT,
    identity_reference=True,
    head_crop_reference=True,
    identity_first=True,
)

MESH_GUIDED_IDFIRST_VIEW_RIGHT_V1 = ImagePreparationProfile(
    id="mesh-guided-idfirst-view-right-v1",
    version=1,
    prompt=(
        _MESH_GUIDED_IDFIRST_INTRO
        + _MESH_GUIDED_RIGHT_SENTENCE
        + _MESH_GUIDED_PROMPT_BODY
    ),
    negative_prompt=_MESH_GUIDED_NEGATIVE_PROMPT,
    identity_reference=True,
    head_crop_reference=True,
    identity_first=True,
)

MESH_GUIDED_IDFIRST_VIEW_FRONT_LEFT_V1 = ImagePreparationProfile(
    id="mesh-guided-idfirst-view-front-left-v1",
    version=1,
    prompt=(
        _MESH_GUIDED_IDFIRST_INTRO
        + _MESH_GUIDED_FRONT_LEFT_SENTENCE
        + _MESH_GUIDED_PROMPT_BODY
    ),
    negative_prompt=_MESH_GUIDED_NEGATIVE_PROMPT,
    identity_reference=True,
    head_crop_reference=True,
    identity_first=True,
)

MESH_GUIDED_IDFIRST_VIEW_FRONT_RIGHT_V1 = ImagePreparationProfile(
    id="mesh-guided-idfirst-view-front-right-v1",
    version=1,
    prompt=(
        _MESH_GUIDED_IDFIRST_INTRO
        + _MESH_GUIDED_FRONT_RIGHT_SENTENCE
        + _MESH_GUIDED_PROMPT_BODY
    ),
    negative_prompt=_MESH_GUIDED_NEGATIVE_PROMPT,
    identity_reference=True,
    head_crop_reference=True,
    identity_first=True,
)

MESH_GUIDED_IDFIRST_VIEW_BACK_LEFT_V1 = ImagePreparationProfile(
    id="mesh-guided-idfirst-view-back-left-v1",
    version=1,
    prompt=(
        _MESH_GUIDED_IDFIRST_INTRO
        + _MESH_GUIDED_BACK_LEFT_SENTENCE
        + _MESH_GUIDED_PROMPT_BODY
        + " "
        + _MESH_GUIDED_BALD_SCALP_SENTENCES
    ),
    negative_prompt=_MESH_GUIDED_NEGATIVE_PROMPT,
    identity_reference=True,
    identity_first=True,
)

MESH_GUIDED_IDFIRST_VIEW_BACK_RIGHT_V1 = ImagePreparationProfile(
    id="mesh-guided-idfirst-view-back-right-v1",
    version=1,
    prompt=(
        _MESH_GUIDED_IDFIRST_INTRO
        + _MESH_GUIDED_BACK_RIGHT_SENTENCE
        + _MESH_GUIDED_PROMPT_BODY
        + " "
        + _MESH_GUIDED_BALD_SCALP_SENTENCES
    ),
    negative_prompt=_MESH_GUIDED_NEGATIVE_PROMPT,
    identity_reference=True,
    identity_first=True,
)

MESH_GUIDED_IDFIRST_VIEW_SYNTHESIS_PROFILES: dict[str, ImagePreparationProfile] = {
    "front": MESH_GUIDED_IDFIRST_VIEW_FRONT_V1,
    "back": MESH_GUIDED_IDFIRST_VIEW_BACK_V1,
    "left": MESH_GUIDED_IDFIRST_VIEW_LEFT_V1,
    "right": MESH_GUIDED_IDFIRST_VIEW_RIGHT_V1,
    "front-left": MESH_GUIDED_IDFIRST_VIEW_FRONT_LEFT_V1,
    "front-right": MESH_GUIDED_IDFIRST_VIEW_FRONT_RIGHT_V1,
    "back-left": MESH_GUIDED_IDFIRST_VIEW_BACK_LEFT_V1,
    "back-right": MESH_GUIDED_IDFIRST_VIEW_BACK_RIGHT_V1,
}

def _without_indexed_intro(profile: ImagePreparationProfile) -> ImagePreparationProfile:
    # The V0 wording (2026-09-05 identity levers): the indexed intro cost the
    # front view 0.73 -> 0.16 SSIM against the T-pose front, so the plain
    # family keeps the per-view sentence + body only, everything else equal.
    prompt = profile.prompt
    for intro in (_MESH_GUIDED_INDEXED_INTRO_WITH_HEAD_CROP, _MESH_GUIDED_INDEXED_INTRO):
        if prompt.startswith(intro):
            prompt = prompt[len(intro):]
            break
    return replace(profile, id=profile.id.replace("mesh-guided-", "mesh-guided-plain-", 1), prompt=prompt)


MESH_GUIDED_PLAIN_VIEW_SYNTHESIS_PROFILES: dict[str, ImagePreparationProfile] = {
    view: _without_indexed_intro(profile)
    for view, profile in MESH_GUIDED_VIEW_SYNTHESIS_PROFILES.items()
}

VIEW_SYNTHESIS_PROFILE_FAMILIES: dict[str, dict[str, ImagePreparationProfile]] = {
    "tpose": VIEW_SYNTHESIS_PROFILES,
    "source-pose": SOURCE_POSE_VIEW_SYNTHESIS_PROFILES,
    "mesh-guided": MESH_GUIDED_VIEW_SYNTHESIS_PROFILES,
    "mesh-guided-plain": MESH_GUIDED_PLAIN_VIEW_SYNTHESIS_PROFILES,
    "mesh-guided-idfirst": MESH_GUIDED_IDFIRST_VIEW_SYNTHESIS_PROFILES,
}


def resolve_view_synthesis_profile(
    view: str, family: str = "tpose"
) -> ImagePreparationProfile:
    try:
        profiles = VIEW_SYNTHESIS_PROFILE_FAMILIES[family]
    except KeyError as exc:
        valid = sorted(VIEW_SYNTHESIS_PROFILE_FAMILIES)
        raise ValueError(f"Unsupported view synthesis profile family: {family}; valid: {valid}") from exc
    try:
        return profiles[view]
    except KeyError as exc:
        raise ValueError(f"Unsupported view synthesis label: {view}") from exc


def build_view_synthesis_metadata(
    *,
    base_metadata: dict[str, object],
    comfy_result: dict[str, object],
    view: str,
    source_artifact_id: str,
    profile: ImagePreparationProfile,
    seed: int,
) -> dict[str, object]:
    """The provenance-complete metadata shape stamped on one synthesized-view
    artifact. PIPELINE_POLICY.md's S1 contract requires source views, seed,
    and model recorded on every generated image; this is that record."""
    return {
        **base_metadata,
        **comfy_result,
        "synthesized_view": view,
        "source_artifact_id": source_artifact_id,
        "adapter": "comfyui",
        "view_synthesis_profile": profile.as_dict(),
        "seed": seed,
    }


def resolve_preparation_profile(value: str) -> ImagePreparationProfile:
    selected = value.strip()
    try:
        return PROFILES[selected]
    except KeyError as exc:
        raise ValueError(f"Unsupported image preparation profile: {selected}") from exc


def build_flux2_klein_image_edit(
    *,
    input_name: str,
    pose_input_name: str | None = None,
    identity_reference_name: str | None = None,
    head_crop_reference_name: str | None = None,
    output_prefix: str,
    profile: ImagePreparationProfile,
    seed: int,
    identity_first: bool = False,
) -> dict[str, dict[str, object]]:
    """Build the pinned ComfyUI API graph used for one reviewable pose candidate."""

    workflow = {
        "70": {"class_type": "UNETLoader", "inputs": {
            "unet_name": FLUX2_KLEIN_MODEL,
            "weight_dtype": "default",
        }},
        "71": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen_3_4b.safetensors", "type": "flux2", "device": "default",
        }},
        "72": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "90": {"class_type": "LoadImage", "inputs": {"image": input_name}},
        "74": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["71", 0], "text": profile.prompt,
        }},
        "67": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["71", 0], "text": profile.negative_prompt,
        }},
        "80": {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": ["90", 0], "upscale_method": "nearest-exact",
            "megapixels": 1.0, "resolution_steps": 1,
        }},
        "78": {"class_type": "VAEEncode", "inputs": {
            "pixels": ["80", 0], "vae": ["72", 0],
        }},
        "81": {"class_type": "GetImageSize", "inputs": {"image": ["80", 0]}},
        "66": {"class_type": "EmptyFlux2LatentImage", "inputs": {
            "width": profile.width, "height": profile.height, "batch_size": 1,
        }},
        "73": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "61": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "62": {"class_type": "Flux2Scheduler", "inputs": {
            "steps": 20, "width": profile.width, "height": profile.height,
        }},
        "63": {"class_type": "CFGGuider", "inputs": {
            "model": ["70", 0], "positive": ["74", 0],
            "negative": ["67", 0], "cfg": 5.0,
        }},
        "64": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["73", 0], "guider": ["63", 0], "sampler": ["61", 0],
            "sigmas": ["62", 0], "latent_image": ["66", 0],
        }},
        "65": {"class_type": "VAEDecode", "inputs": {
            "samples": ["64", 0], "vae": ["72", 0],
        }},
        "91": {"class_type": "PreviewImage", "inputs": {"images": ["65", 0]}},
    }
    # Track whichever conditioning node is currently wired into the guider so
    # each reference image chains onto whatever came before it -- the
    # 45ac3e4 chaining pattern, generalized to stack -- rather than always
    # onto the base text conditioning.
    positive_ref = "74"
    negative_ref = "67"

    def chain_reference(
        latent_node_id: str, positive_node_id: str, negative_node_id: str
    ) -> None:
        nonlocal positive_ref, negative_ref
        workflow[positive_node_id] = {"class_type": "ReferenceLatent", "inputs": {
            "conditioning": [positive_ref, 0], "latent": [latent_node_id, 0],
        }}
        # Every reference latent is chained onto the NEGATIVE conditioning
        # too, symmetric with the positive side. Flux Kontext-style
        # ReferenceLatent conditioning is steered by (positive - negative)
        # under CFG; a reference attached to positive only lets CFG amplify
        # whatever is unique to that latent unopposed, swamping the
        # character's identity. Symmetric chaining keeps CFG's differential
        # pointed at the text instruction instead.
        workflow[negative_node_id] = {"class_type": "ReferenceLatent", "inputs": {
            "conditioning": [negative_ref, 0], "latent": [latent_node_id, 0],
        }}
        positive_ref = positive_node_id
        negative_ref = negative_node_id

    def add_identity_nodes() -> None:
        workflow.update({
            "102": {"class_type": "LoadImage", "inputs": {"image": identity_reference_name}},
            "103": {"class_type": "ImageScaleToTotalPixels", "inputs": {
                "image": ["102", 0], "upscale_method": "nearest-exact",
                "megapixels": 1.0, "resolution_steps": 1,
            }},
            "104": {"class_type": "VAEEncode", "inputs": {
                "pixels": ["103", 0], "vae": ["72", 0],
            }},
        })

    if identity_first and identity_reference_name is not None:
        # Identity-first order (mesh-guided-idfirst): the character photo
        # chains onto the base text conditioning FIRST, making it CFG's
        # primary edit image; the clay/source latent (78) chains second, as
        # the structural (silhouette/pose/camera) reference.
        add_identity_nodes()
        chain_reference("104", "76", "77")
        chain_reference("78", "105", "106")
    else:
        chain_reference("78", "76", "77")
        if pose_input_name is not None:
            workflow.update({
                "93": {"class_type": "LoadImage", "inputs": {"image": pose_input_name}},
                "95": {"class_type": "ImageScaleToTotalPixels", "inputs": {
                    "image": ["93", 0], "upscale_method": "nearest-exact",
                    "megapixels": 1.0, "resolution_steps": 1,
                }},
                "94": {"class_type": "VAEEncode", "inputs": {
                    "pixels": ["95", 0], "vae": ["72", 0],
                }},
            })
            chain_reference("94", "92", "96")
        if identity_reference_name is not None:
            # Mesh-guided views (identity NOT first): the identity photo
            # chains after the source/clay latent, as a secondary reference.
            add_identity_nodes()
            chain_reference("104", "105", "106")

    if head_crop_reference_name is not None:
        workflow.update({
            "97": {"class_type": "LoadImage", "inputs": {"image": head_crop_reference_name}},
            "99": {"class_type": "ImageScaleToTotalPixels", "inputs": {
                "image": ["97", 0], "upscale_method": "nearest-exact",
                "megapixels": 0.25, "resolution_steps": 1,
            }},
            "98": {"class_type": "VAEEncode", "inputs": {
                "pixels": ["99", 0], "vae": ["72", 0],
            }},
        })
        # Bead ncxzd's mechanism: a tight head/shoulders crop of the
        # reference, chained last (onto whatever came before it) so it is
        # the final, closest-anchor override for the head/skull region.
        chain_reference("98", "100", "101")

    workflow["63"]["inputs"]["positive"] = [positive_ref, 0]
    workflow["63"]["inputs"]["negative"] = [negative_ref, 0]

    return workflow


def detect_subject_bbox(image_bytes: bytes) -> tuple[int, int, int, int]:
    """Estimate the (left, top, right, bottom) pixel bbox of the non-background
    subject in a reference image, by comparing every pixel against a
    background color sampled from the four corners -- this pipeline's
    reference images are consistently rendered on a plain, near-uniform
    background (see every profile's "plain neutral background" requirement),
    so corner sampling is a reliable, dependency-free background estimate."""
    import numpy as np
    from io import BytesIO
    from PIL import Image

    with Image.open(BytesIO(image_bytes)) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.int16)
    height, width, _ = rgb.shape
    corners = np.array(
        [rgb[0, 0], rgb[0, width - 1], rgb[height - 1, 0], rgb[height - 1, width - 1]]
    )
    background = corners.mean(axis=0)
    deviation = np.abs(rgb - background).sum(axis=2)
    mask = deviation > 24
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return (0, 0, width, height)
    return (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)


def build_head_crop_reference(image_bytes: bytes, *, crop_fraction: float = 0.35) -> bytes:
    """Crop the top `crop_fraction` of the subject's own bounding box (the
    head and upper shoulders) out of a reference image, re-encoded as PNG
    bytes. This is the head-crop escalation named in the back-view v4
    provenance comment above `VIEW_SYNTHESIS_BACK_V4`: a second
    ReferenceLatent built from this crop gives the model an unmistakable,
    hard-to-ignore close-up of exactly the region (the skull) it otherwise
    has license to invent when redrawing an unseen viewpoint."""
    from io import BytesIO
    from PIL import Image

    left, top, right, bottom = detect_subject_bbox(image_bytes)
    crop_bottom = top + max(1, round((bottom - top) * crop_fraction))
    with Image.open(BytesIO(image_bytes)) as opened:
        cropped = opened.convert("RGB").crop((left, top, right, crop_bottom))
        buffer = BytesIO()
        cropped.save(buffer, format="PNG")
        return buffer.getvalue()


COMFYUI_SERVICE_UNIT = "comfyui.service"
COMFYUI_START_TIMEOUT_S = 180.0


def comfyui_reachable(service: str, *, timeout: float = 2.0) -> bool:
    try:
        return requests.get(f"{service}/system_stats", timeout=timeout).ok
    except requests.RequestException:
        return False


def ensure_comfyui(
    service: str,
    *,
    start: Callable[[], None] | None = None,
    reachable: Callable[[str], bool] = comfyui_reachable,
    sleep: Callable[[float], None] = time.sleep,
    timeout: float = COMFYUI_START_TIMEOUT_S,
) -> None:
    """Bring ComfyUI up before the first request to it.

    On the GPU box ComfyUI is the outbound pool's on-demand systemd user unit:
    its idle watcher stops it ~60 s after the queue empties, so a studio job
    that starts after a quiet spell finds :8188 refusing connections
    (create-tpose-set failed that way on 2026-09-06). Starting the unit is
    idempotent; polling waits for the server to answer.
    """
    if reachable(service):
        return
    if start is None:
        import subprocess

        def start() -> None:
            subprocess.run(["systemctl", "--user", "start", COMFYUI_SERVICE_UNIT], check=False, timeout=30)
    try:
        start()
    except OSError as exc:
        # No systemctl on this box (macOS, CI, a container): there is no unit
        # to start, so this is the plain "ComfyUI is not running" condition.
        # Raise it as the same ConnectionError the callers already translate
        # into their "ComfyUI unavailable" message, rather than leaking a
        # FileNotFoundError from the shell-out.
        raise requests.exceptions.ConnectionError(
            f"ComfyUI at {service} is not reachable and {COMFYUI_SERVICE_UNIT} "
            f"could not be started ({exc}). Start ComfyUI manually before retrying."
        ) from exc
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if reachable(service):
            return
        sleep(3.0)
    raise RuntimeError(f"ComfyUI at {service} did not come up within {timeout:.0f}s after starting {COMFYUI_SERVICE_UNIT}")


def run_comfyui_image_edit(
    source: Path,
    output: Path,
    *,
    profile: ImagePreparationProfile,
    seed: int,
    base_url: str,
    session: Any = requests,
    timeout_seconds: float = 300.0,
    on_accepted: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    pose_guide_root: Path | None = None,
    identity_image: Path | None = None,
) -> dict[str, object]:
    """Run one pinned ComfyUI candidate and retain its reproducibility metadata."""

    if profile.identity_reference and identity_image is None:
        raise ValueError(
            f"Profile {profile.id!r} requires identity_reference=True but no identity_image was given"
        )
    service = base_url.rstrip("/")
    if session is requests:  # a live server; an injected session is a test double with no unit to start
        ensure_comfyui(service)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    upload_name = f"aismith-preparation-{source_digest}{source.suffix.lower() or '.png'}"
    with source.open("rb") as image_file:
        upload = session.post(
            f"{service}/upload/image",
            files={"image": (upload_name, image_file, "application/octet-stream")},
            data={"overwrite": "true"},
            timeout=30,
        )
    upload.raise_for_status()
    input_name = str(upload.json()["name"])
    pose_input_name: str | None = None
    pose_guide_sha256: str | None = None
    if profile.pose_guide_asset:
        asset_root = pose_guide_root or Path(__file__).parent
        pose_guide = asset_root / profile.pose_guide_asset
        if not pose_guide.is_file():
            raise FileNotFoundError(f"Pose guide asset is missing: {profile.pose_guide_asset}")
        pose_bytes = pose_guide.read_bytes()
        pose_guide_sha256 = hashlib.sha256(pose_bytes).hexdigest()
        pose_upload_name = f"aismith-pose-guide-{pose_guide_sha256[:16]}{pose_guide.suffix.lower()}"
        with pose_guide.open("rb") as pose_file:
            pose_upload = session.post(
                f"{service}/upload/image",
                files={"image": (pose_upload_name, pose_file, "application/octet-stream")},
                data={"overwrite": "true"},
                timeout=30,
            )
        pose_upload.raise_for_status()
        pose_input_name = str(pose_upload.json()["name"])
    identity_reference_name: str | None = None
    identity_bytes: bytes | None = None
    identity_sha256: str | None = None
    if profile.identity_reference:
        identity_bytes = identity_image.read_bytes()
        identity_sha256 = hashlib.sha256(identity_bytes).hexdigest()
        identity_upload_name = (
            f"aismith-identity-reference-{identity_sha256[:16]}{identity_image.suffix.lower() or '.png'}"
        )
        identity_upload = session.post(
            f"{service}/upload/image",
            files={"image": (identity_upload_name, identity_bytes, "application/octet-stream")},
            data={"overwrite": "true"},
            timeout=30,
        )
        identity_upload.raise_for_status()
        identity_reference_name = str(identity_upload.json()["name"])
    head_crop_reference_name: str | None = None
    head_crop_sha256: str | None = None
    if profile.head_crop_reference:
        # Mesh-guided profiles: the source is a headless clay render, so the
        # head crop must come from the IDENTITY photo instead.
        head_crop_source_bytes = identity_bytes if profile.identity_reference else source.read_bytes()
        head_crop_bytes = build_head_crop_reference(head_crop_source_bytes)
        head_crop_sha256 = hashlib.sha256(head_crop_bytes).hexdigest()
        head_crop_upload_name = f"aismith-head-crop-{head_crop_sha256[:16]}.png"
        head_crop_upload = session.post(
            f"{service}/upload/image",
            files={"image": (head_crop_upload_name, head_crop_bytes, "image/png")},
            data={"overwrite": "true"},
            timeout=30,
        )
        head_crop_upload.raise_for_status()
        head_crop_reference_name = str(head_crop_upload.json()["name"])
    workflow = build_flux2_klein_image_edit(
        input_name=input_name,
        pose_input_name=pose_input_name,
        identity_reference_name=identity_reference_name,
        head_crop_reference_name=head_crop_reference_name,
        output_prefix=f"aismith/{output.stem}",
        profile=profile,
        seed=seed,
        identity_first=profile.identity_first,
    )
    queued = session.post(f"{service}/prompt", json={"prompt": workflow}, timeout=30)
    queued_payload = queued.json()
    node_errors = queued_payload.get("node_errors") or {}
    if node_errors:
        raise RuntimeError(f"ComfyUI rejected the image preparation graph: {node_errors}")
    queued.raise_for_status()
    prompt_id = str(queued_payload["prompt_id"])
    if on_accepted:
        on_accepted(prompt_id)

    def cancel_owned_prompt() -> None:
        queue = session.get(f"{service}/queue", timeout=30)
        queue.raise_for_status()
        queue_payload = queue.json()
        running_ids = {
            str(item[1])
            for item in queue_payload.get("queue_running", [])
            if isinstance(item, list) and len(item) > 1
        }
        session.post(
            f"{service}/queue", json={"delete": [prompt_id]}, timeout=30,
        ).raise_for_status()
        if prompt_id in running_ids:
            session.post(f"{service}/interrupt", json={}, timeout=30).raise_for_status()

    deadline = time.monotonic() + timeout_seconds
    execution: dict[str, object] | None = None
    while time.monotonic() < deadline:
        if cancelled and cancelled():
            cancel_owned_prompt()
            raise InterruptedError(f"ComfyUI prompt {prompt_id} was cancelled")
        history_response = session.get(f"{service}/history/{prompt_id}", timeout=30)
        history_response.raise_for_status()
        execution = history_response.json().get(prompt_id)
        if execution and execution.get("status", {}).get("completed"):
            break
        time.sleep(0.5)
    if not execution or not execution.get("status", {}).get("completed"):
        cancel_owned_prompt()
        raise TimeoutError(f"ComfyUI image preparation timed out after {timeout_seconds:g} seconds")
    status = execution.get("status", {})
    if status.get("status_str") != "success":
        raise RuntimeError(f"ComfyUI image preparation failed: {status}")

    images = [
        image
        for node in execution.get("outputs", {}).values()
        for image in node.get("images", [])
    ]
    if len(images) != 1:
        raise RuntimeError(
            f"ComfyUI preparation requires exactly one image output; received {len(images)}"
        )
    generated = images[0]
    download = session.get(
        f"{service}/view",
        params={
            "filename": generated["filename"],
            "subfolder": generated.get("subfolder", ""),
            "type": generated.get("type", "output"),
        },
        timeout=30,
    )
    download.raise_for_status()
    output.write_bytes(download.content)
    metadata = {
        "prompt_id": prompt_id,
        "workflow_version": (
            FLUX2_KLEIN_MULTI_REFERENCE_WORKFLOW_VERSION
            if (pose_guide_sha256 or identity_sha256 or head_crop_sha256)
            else FLUX2_KLEIN_WORKFLOW_VERSION
        ),
        "model": FLUX2_KLEIN_MODEL,
        "text_encoder": "qwen_3_4b.safetensors",
        "vae": "flux2-vae.safetensors",
        "comfyui_url": service,
        "output_dimensions": [profile.width, profile.height],
        "reference_scaling": "aspect-preserving-one-megapixel",
    }
    if pose_guide_sha256:
        metadata["pose_guide_asset"] = profile.pose_guide_asset
        metadata["pose_guide_sha256"] = pose_guide_sha256
    if identity_sha256:
        metadata["identity_reference"] = True
        metadata["identity_reference_sha256"] = identity_sha256
    if head_crop_sha256:
        metadata["head_crop_reference"] = True
        metadata["head_crop_sha256"] = head_crop_sha256
    return metadata
