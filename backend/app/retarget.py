from __future__ import annotations


SOMA_TO_ROLE = {
    "Hips": "pelvis",
    "Spine1": "spine",
    "Spine2": "spine",
    "Chest": "spine",
    "Neck1": "head",
    "Head": "head",
    "LeftShoulder": "left_upper_arm",
    "LeftArm": "left_upper_arm",
    "LeftForeArm": "left_lower_arm",
    "LeftHand": "left_hand",
    "RightShoulder": "right_upper_arm",
    "RightArm": "right_upper_arm",
    "RightForeArm": "right_lower_arm",
    "RightHand": "right_hand",
    "LeftLeg": "left_upper_leg",
    "LeftShin": "left_lower_leg",
    "LeftFoot": "left_foot",
    "RightLeg": "right_upper_leg",
    "RightShin": "right_lower_leg",
    "RightFoot": "right_foot",
}


# Each source bone lists its fallback target-name candidates in priority
# order. UniRig's own naming (spine_1/spine_2/spine_3, neck) comes first;
# the older mesh2motion-style names (spine_01.., neck_01) are kept as
# lower-priority candidates for rigs that still use them. Where a candidate
# is absent from the target skeleton entirely (e.g. no separate clavicle
# bone), resolution falls through to the literal source name and then to
# the certified role target -- see ROLE_FIRST_SOURCES below.
SOMA_TARGET_FALLBACKS: dict[str, tuple[str, ...]] = {
    "Root": ("root",), "Hips": ("pelvis",),
    "Spine1": ("spine_1", "spine_01"),
    "Spine2": ("spine_2", "spine_02"),
    "Chest": ("spine_3", "spine_03"),
    "Neck1": ("neck", "neck_01"),
    "Head": ("head",), "LeftShoulder": ("clavicle_l",),
    "LeftArm": ("upperarm_l",), "LeftForeArm": ("lowerarm_l",),
    "LeftHand": ("hand_l",), "LeftHandMiddle1": ("middle_01_l",),
    "RightShoulder": ("clavicle_r",), "RightArm": ("upperarm_r",),
    "RightForeArm": ("lowerarm_r",), "RightHand": ("hand_r",),
    "RightHandMiddle1": ("middle_01_r",), "LeftLeg": ("thigh_l",),
    "LeftShin": ("calf_l",), "LeftFoot": ("foot_l",), "LeftToeBase": ("ball_l",),
    "RightLeg": ("thigh_r",), "RightShin": ("calf_r",),
    "RightFoot": ("foot_r",), "RightToeBase": ("ball_r",),
}

ROLE_FIRST_SOURCES = {
    "LeftLeg", "LeftShin", "LeftFoot",
    "RightLeg", "RightShin", "RightFoot",
}

# When the target skeleton lacks a bone for every SOMA segment (e.g. one
# "spine" bone for Spine1/Spine2/Chest, or no separate clavicle), several
# source bones resolve to the very same target -- see
# resolve_motion_mapping's docstring. select_deepest_source_per_target
# picks the single most-distal source to actually drive that target's
# rotation each frame, rather than an arbitrary last-writer-wins.
SOMA_SOURCE_DEPTH = {
    "Hips": 0,
    "Spine1": 1, "Spine2": 2, "Chest": 3, "Neck1": 4, "Head": 5,
    "LeftShoulder": 1, "LeftArm": 2, "LeftForeArm": 3, "LeftHand": 4,
    "LeftHandMiddle1": 5,
    "RightShoulder": 1, "RightArm": 2, "RightForeArm": 3, "RightHand": 4,
    "RightHandMiddle1": 5,
    "LeftLeg": 1, "LeftShin": 2, "LeftFoot": 3, "LeftToeBase": 4,
    "RightLeg": 1, "RightShin": 2, "RightFoot": 3, "RightToeBase": 4,
}


def resolve_motion_mapping(
    source_names: set[str],
    target_names: set[str],
    resolved_roles: dict[str, str],
) -> dict[str, str]:
    """Map every present SOMA source bone to a target bone name.

    Multiple source bones legitimately collapse onto the same target when
    the target skeleton has fewer segments than SOMA does (a single "spine"
    bone standing in for Spine1/Spine2/Chest, or no clavicle). That is
    intentional here -- every source bone is still reported so callers know
    which SOMA joints are covered -- but only one of the colliding sources
    should actually drive the shared target's rotation each frame; see
    `select_deepest_source_per_target`.
    """
    target_by_casefold = {name.casefold(): name for name in target_names}
    mapping: dict[str, str] = {}
    for source_name, fallbacks in SOMA_TARGET_FALLBACKS.items():
        if source_name not in source_names:
            continue
        role = SOMA_TO_ROLE.get(source_name)
        role_target = resolved_roles.get(role, "") if role else ""
        if source_name in ROLE_FIRST_SOURCES:
            candidates = [role_target, *fallbacks, source_name]
        else:
            candidates = [*fallbacks, source_name, role_target]
        target_name = next(
            (
                target_by_casefold[name.casefold()]
                for name in candidates
                if name and name.casefold() in target_by_casefold
            ),
            None,
        )
        if target_name:
            mapping[source_name] = target_name
    return mapping


def select_deepest_source_per_target(mapping: dict[str, str]) -> dict[str, str]:
    """Invert a (possibly many-to-one) source->target mapping to a
    one-to-one target->source mapping, keeping the most distal source bone
    (deepest in the SOMA hierarchy) whenever several sources collapse onto
    the same target -- e.g. Chest over Spine2 over Spine1, Head over Neck1,
    LeftArm over LeftShoulder. Sources with no known depth are treated as
    shallowest (depth 0) and lose ties to any source with a known depth.
    """
    chosen: dict[str, str] = {}
    chosen_depth: dict[str, int] = {}
    for source_name, target_name in mapping.items():
        depth = SOMA_SOURCE_DEPTH.get(source_name, 0)
        if target_name not in chosen or depth > chosen_depth[target_name]:
            chosen[target_name] = source_name
            chosen_depth[target_name] = depth
    return chosen


def select_character_armature(
    armature_bones: dict[str, set[str]],
    resolved_roles: dict[str, str],
) -> str:
    if len(armature_bones) == 1:
        return next(iter(armature_bones))
    certified = {name.casefold() for name in resolved_roles.values() if name}
    scores = {
        name: len(certified & {bone.casefold() for bone in bones})
        for name, bones in armature_bones.items()
    }
    best_score = max(scores.values(), default=0)
    winners = [name for name, score in scores.items() if score == best_score]
    if best_score == 0 or len(winners) != 1:
        raise ValueError("could not uniquely identify the certified character armature")
    return winners[0]


def select_motion_translation_bones(
    source_names: set[str],
    mapping: dict[str, str],
) -> tuple[str, str]:
    """Select the BVH position-channel owner and its mapped target bone."""
    if "Root" in source_names and "Root" in mapping:
        return "Root", mapping["Root"]
    if "Hips" in source_names and "Hips" in mapping:
        return "Hips", mapping["Hips"]
    raise ValueError("motion mapping has no translated root or hips bone")
