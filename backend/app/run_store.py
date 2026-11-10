from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Literal, NamedTuple

from filelock import FileLock


class ReviewState(str, Enum):
    UNREVIEWED = "unreviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


# A run is a general 2D->3D pipeline; biped rigging is one optional path
# through it. "auto" is the historical/default behaviour -- every stage stays
# available and rig-stage routing falls back to inferring from the rig
# certificate profile ("humanoid" vs "chain") once one exists.
SubjectKind = Literal["auto", "object", "creature", "biped"]

# Stage ids from the frontend Character Journey (frontend/src/journey.ts).
_ALL_STAGE_IDS: tuple[str, ...] = ("prepare", "generate", "clean", "skeleton", "animate", "review")
# An `object` run has no skeleton to predict, no motion to attach, and no
# animated-review gate to clear -- those three stages are not applicable.
_OBJECT_STAGE_IDS: tuple[str, ...] = ("prepare", "generate", "clean")


# Stages whose completed output is an approvable T-pose FRONT. `prepare-image`
# is the original single-step form; `create-tpose-set` (bead 4vquq) is the
# composite that draws the same front and then fans the three views off it.
# Anything asking "has this run produced a front the operator can approve, and
# which one is newest?" reads this tuple rather than one literal stage name.
PREPARED_FRONT_STAGES: tuple[str, ...] = ("prepare-image", "create-tpose-set")


def applicable_stages_for(subject_kind: str) -> list[str]:
    """Journey stage ids applicable to a run of the given subject kind."""
    if subject_kind == "object":
        return list(_OBJECT_STAGE_IDS)
    return list(_ALL_STAGE_IDS)


@dataclass
class StageAttempt:
    id: str
    stage: str
    status: str
    input_artifact_id: str
    settings: dict[str, Any]
    job_id: str | None = None
    retry_of_attempt_id: str | None = None
    promoted: bool = True
    output_artifact_id: str | None = None
    logs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None


@dataclass
class HumanCheckpoint:
    note: str
    artifact_id: str
    completed_at: float = field(default_factory=time.time)


@dataclass
class RunManifest:
    id: str
    name: str
    status: str
    source_artifact_id: str
    source_artifact_kind: str = "unknown"
    current_artifact_id: str | None = None
    review_status: ReviewState = ReviewState.UNREVIEWED
    review_note: str = ""
    reviewed_artifact_id: str | None = None
    subject_kind: SubjectKind = "auto"
    applicable_stages: list[str] = field(default_factory=lambda: applicable_stages_for("auto"))
    checkpoints: dict[str, HumanCheckpoint] = field(default_factory=dict)
    attempts: list[StageAttempt] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


def summarize_run(run: RunManifest) -> RunManifest:
    """Return the lifecycle fields needed by the history index without payload logs."""
    attempts = [
        replace(
            attempt,
            settings={},
            logs=[],
            warnings=[],
            error=None,
        )
        for attempt in run.attempts
    ]
    return replace(run, attempts=attempts)


class ViewSynthesisSource(NamedTuple):
    """Which image a `synthesize-views` job will redraw from, and why.

    `reason` is prose written for the operator reading an artifact's
    metadata months later; it names the artifacts involved so the choice can
    be audited without replaying the run.
    """

    artifact_id: str
    label: str
    reason: str


def select_view_synthesis_source(
    run: RunManifest,
    requested_source_artifact_id: str,
    settings: dict[str, Any],
) -> ViewSynthesisSource:
    """Pick the reference image for a `synthesize-views` job (beads t0z17, 8c9gf).

    Priority order:
      1. An explicit `settings["source_artifact_id"]` override always wins --
         a caller that deliberately names a source (e.g. re-synthesizing
         from the raw upload even though an approved T-pose candidate
         exists) is never second-guessed.
      2. The run's NEWEST APPROVED prepare-image candidate: the most recent
         `prepare-image` attempt that completed and was promoted. Redrawing
         back/left/right FROM an image that already holds the target pose and
         the reviewed outfit is far more reliable than redrawing from a raw
         front photo that may be in an arbitrary pose -- the fix for the
         arms-at-sides regression the t0z17 verification run found.
      3. Otherwise, the raw front image the caller passed as
         `source_artifact_id`.

    Rule 2 deliberately does NOT require the candidate to still be the run's
    `current_artifact_id`, and that is bead 8c9gf's whole point. `promoted`
    on a prepare-image attempt already means "an operator approved this
    candidate" -- prepare-image lands with `promoted=False` and only
    `promote_artifact` flips it. The current-artifact pointer means something
    else entirely: it is the run's LATEST asset, and Generate, Clean, and
    rigging each advance it to a GLB. Gating on the pointer therefore made
    the approved reference expire the moment the run moved on, so a
    re-synthesis after Generate silently fell back to the raw upload. Live
    consequence (run 39247f85, "lyra"): the raw upload is a robed witch, the
    approved candidate is a purple jumpsuit, and the fallback produced views
    faithful to the wrong source -- which TRELLIS.2 then reconciled into a
    fused-leg skirt mesh. A later GLB is not a view-synthesis reference; the
    approved image is, for as long as the run keeps it.

    Returns a `ViewSynthesisSource`. `label` is the machine-readable
    provenance tag stamped on every synthesized view's metadata -- one of
    "explicit-override", "approved-t-pose-candidate", or "front-image" --
    and `reason` is the human-readable justification stamped beside it.
    """
    override = settings.get("source_artifact_id")
    if isinstance(override, str) and override.strip():
        chosen = override.strip()
        return ViewSynthesisSource(
            chosen,
            "explicit-override",
            f"settings.source_artifact_id explicitly named {chosen}, so that "
            f"artifact is the synthesis reference; an explicitly named source is "
            f"never second-guessed.",
        )

    approved_preparation = next(
        (
            attempt
            for attempt in reversed(run.attempts)
            if attempt.stage in PREPARED_FRONT_STAGES
            and attempt.status == "completed"
            and attempt.promoted
            and attempt.output_artifact_id
        ),
        None,
    )
    if approved_preparation is not None and approved_preparation.output_artifact_id:
        chosen = approved_preparation.output_artifact_id
        if chosen == run.current_artifact_id:
            reason = (
                f"{chosen} is this run's newest approved prepare-image candidate "
                f"and is still the run's current artifact, so views are redrawn "
                f"from the reviewed pose and outfit."
            )
        else:
            reason = (
                f"{chosen} is this run's newest approved prepare-image candidate. "
                f"The run's current artifact has since advanced to "
                f"{run.current_artifact_id}, but a downstream mesh is not a "
                f"view-synthesis reference and the raw front image "
                f"{requested_source_artifact_id} may not depict the approved "
                f"outfit or pose, so the approved candidate still wins."
            )
        return ViewSynthesisSource(chosen, "approved-t-pose-candidate", reason)

    return ViewSynthesisSource(
        requested_source_artifact_id,
        "front-image",
        f"this run has no approved prepare-image candidate, so the caller's front "
        f"image {requested_source_artifact_id} is the only reference available.",
    )


class RunStore:
    _RIG_STAGES = {"rig", "auto-rig", "rig-adjust", "rig-transfer", "target-proxy-rig", "repair-weights"}
    # Stages whose completed output sits DOWNSTREAM of the skeleton: motion
    # baked onto an already-approved rig. Approving a skeleton must never
    # rewind the run's current artifact past one of these.
    _ANIMATED_STAGES = {"generate-motion", "retarget-motion", "apply-clip"}

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[RunManifest]:
        runs = [self._load(path) for path in self.root.glob("*.json")]
        return sorted(runs, key=lambda run: run.created_at, reverse=True)

    def create(
        self,
        name: str,
        source_artifact_id: str,
        source_artifact_kind: str = "unknown",
        subject_kind: SubjectKind = "auto",
    ) -> RunManifest:
        run = RunManifest(
            id=uuid.uuid4().hex,
            name=name,
            status="ready",
            source_artifact_id=source_artifact_id,
            source_artifact_kind=source_artifact_kind,
            current_artifact_id=source_artifact_id,
            subject_kind=subject_kind,
            applicable_stages=applicable_stages_for(subject_kind),
        )
        with self._lock(run.id):
            self._save(run)
        return run

    def get(self, run_id: str) -> RunManifest:
        with self._lock(run_id):
            return self._get_unlocked(run_id)

    def set_subject_kind(self, run_id: str, subject_kind: SubjectKind) -> RunManifest:
        with self._lock(run_id):
            run = self._get_unlocked(run_id)
            run.subject_kind = subject_kind
            run.applicable_stages = applicable_stages_for(subject_kind)
            run.updated_at = time.time()
            self._save(run)
            return run

    def start_attempt(
        self,
        run_id: str,
        stage: str,
        input_artifact_id: str,
        settings: dict[str, Any],
    ) -> StageAttempt:
        with self._lock(run_id):
            run = self._get_unlocked(run_id)
            if input_artifact_id not in self._artifact_ids(run):
                raise ValueError("Input artifact does not belong to this run")
            attempt = StageAttempt(
                id=uuid.uuid4().hex,
                stage=stage,
                status="running",
                input_artifact_id=input_artifact_id,
                settings=settings,
            )
            run.attempts.append(attempt)
            run.status = self._aggregate_status(run)
            run.updated_at = time.time()
            self._save(run)
            return attempt

    def reserve_attempt(
        self,
        run_id: str,
        job_id: str,
        stage: str,
        input_artifact_id: str,
        settings: dict[str, Any],
        retry_of_attempt_id: str | None = None,
    ) -> StageAttempt:
        with self._lock(run_id):
            run = self._get_unlocked(run_id)
            existing = next((item for item in run.attempts if item.job_id == job_id), None)
            if existing is not None:
                if (
                    existing.stage == stage
                    and existing.input_artifact_id == input_artifact_id
                    and existing.settings == settings
                    and existing.retry_of_attempt_id == retry_of_attempt_id
                ):
                    return existing
                raise ValueError(f"Job {job_id} is already associated with another attempt")
            if input_artifact_id not in self._artifact_ids(run):
                raise ValueError("Input artifact does not belong to this run")
            attempt = StageAttempt(
                id=uuid.uuid4().hex,
                stage=stage,
                status="queued",
                input_artifact_id=input_artifact_id,
                settings=settings,
                job_id=job_id,
                retry_of_attempt_id=retry_of_attempt_id,
            )
            run.attempts.append(attempt)
            run.status = self._aggregate_status(run)
            run.updated_at = time.time()
            self._save(run)
            return attempt

    def mark_attempt_running(self, run_id: str, attempt_id: str, job_id: str) -> StageAttempt:
        with self._lock(run_id):
            run = self._get_unlocked(run_id)
            attempt = next((item for item in run.attempts if item.id == attempt_id), None)
            if attempt is None or attempt.job_id != job_id:
                raise KeyError(attempt_id)
            if attempt.status == "running":
                return attempt
            if attempt.status != "queued":
                raise ValueError(f"Attempt {attempt_id} is already {attempt.status}")
            attempt.status = "running"
            run.status = "running"
            run.updated_at = time.time()
            self._save(run)
            return attempt

    def finish_attempt(
        self,
        run_id: str,
        attempt_id: str,
        status: str,
        *,
        output_artifact_id: str | None = None,
        error: str | None = None,
        logs: list[str] | None = None,
        warnings: list[str] | None = None,
        promote_output: bool = True,
    ) -> RunManifest:
        next_logs = list(logs or [])
        next_warnings = list(warnings or [])
        with self._lock(run_id):
            run = self._get_unlocked(run_id)
            attempt = next((item for item in run.attempts if item.id == attempt_id), None)
            if attempt is None:
                raise KeyError(attempt_id)
            if attempt.status not in {"queued", "running"}:
                duplicate = (
                    attempt.status == status
                    and attempt.output_artifact_id == output_artifact_id
                    and attempt.error == error
                    and attempt.logs == next_logs
                    and attempt.warnings == next_warnings
                    and attempt.promoted == promote_output
                )
                if duplicate:
                    return run
                raise ValueError(f"Attempt {attempt_id} is already {attempt.status}")
            attempt.status = status
            attempt.output_artifact_id = output_artifact_id
            attempt.error = error
            attempt.logs = next_logs
            attempt.warnings = next_warnings
            attempt.promoted = promote_output
            attempt.completed_at = time.time()
            if output_artifact_id and promote_output:
                run.current_artifact_id = output_artifact_id
            run.status = self._aggregate_status(run, terminal_fallback=status)
            run.updated_at = attempt.completed_at
            self._save(run)
            return run

    def promote_artifact(self, run_id: str, artifact_id: str, reason: str) -> RunManifest:
        with self._lock(run_id):
            run = self._get_unlocked(run_id)
            if artifact_id not in self._artifact_ids(run):
                raise ValueError("Artifact does not belong to this run")
            self._pin_current_unlocked(run, artifact_id, reason)
            self._save(run)
            return run

    def _pin_current_unlocked(self, run: RunManifest, artifact_id: str, reason: str) -> None:
        """Make `artifact_id` the run's current artifact, and say why in the manifest.

        Every pointer move outside `finish_attempt` goes through here so the
        manifest carries an auditable `promote-artifact` attempt naming the
        reason -- the pointer is never moved silently. Caller holds the lock
        and saves.
        """
        now = time.time()
        producing_attempt = next(
            (
                attempt
                for attempt in reversed(run.attempts)
                if attempt.status == "completed"
                and attempt.stage != "promote-artifact"
                and attempt.output_artifact_id == artifact_id
            ),
            None,
        )
        if producing_attempt is not None:
            producing_attempt.promoted = True
        attempt = StageAttempt(
            id=uuid.uuid4().hex,
            stage="promote-artifact",
            status="completed",
            input_artifact_id=run.current_artifact_id or run.source_artifact_id,
            output_artifact_id=artifact_id,
            settings={"reason": reason},
            promoted=True,
            created_at=now,
            completed_at=now,
        )
        run.attempts.append(attempt)
        run.current_artifact_id = artifact_id
        run.status = self._aggregate_status(run, terminal_fallback="completed")
        run.updated_at = now

    @staticmethod
    def _artifact_ids(run: RunManifest) -> set[str]:
        artifacts = {run.source_artifact_id}
        artifacts.update(item.output_artifact_id for item in run.attempts if item.output_artifact_id)
        return artifacts

    @staticmethod
    def _aggregate_status(run: RunManifest, terminal_fallback: str = "ready") -> str:
        if any(item.status == "running" for item in run.attempts):
            return "running"
        if any(item.status == "queued" for item in run.attempts):
            return "queued"
        return terminal_fallback

    def review(self, run_id: str, status: ReviewState, note: str, artifact_id: str) -> RunManifest:
        with self._lock(run_id):
            run = self._get_unlocked(run_id)
            candidate = next((
                attempt for attempt in reversed(run.attempts)
                if attempt.stage == "retarget-motion"
                and attempt.status == "completed"
                and attempt.output_artifact_id == artifact_id
            ), None)
            if candidate is None:
                raise ValueError("Final review requires a completed animated artifact from this run")
            run.review_status = status
            run.review_note = note
            run.reviewed_artifact_id = artifact_id
            if status == ReviewState.ACCEPTED:
                run.current_artifact_id = artifact_id
            elif run.current_artifact_id == artifact_id:
                run.current_artifact_id = candidate.input_artifact_id
            run.updated_at = time.time()
            self._save(run)
            return run

    def hand_back(self, run_id: str, checkpoint: str, note: str, artifact_id: str) -> RunManifest:
        """Record the operator's durable approval of one exact rig artifact.

        The checkpoint is bound to the artifact the review card displayed and
        promised to approve -- NOT to the run's current-artifact pointer. The
        two are routinely different, for reasons that have nothing to do with
        this approval:

        * A fresh auto-rig whose certificate lands `needs_review` is
          deliberately *retained* (`_rig_candidate_can_promote` in main.py),
          so `finish_attempt` never advances the pointer onto it. Demanding
          `current == artifact` therefore made approval impossible for
          exactly the rigs that most need a human decision -- the operator's
          "I can't even approve when the auto-rig is successful".
        * A later `promote-artifact` (e.g. re-approving the prepared image
          before a re-generate) can move the pointer somewhere else entirely.

        What stays strict is identity: `artifact_id` must be a completed
        rig-stage output of THIS run.

        Approval then *pins* the approved rig as the run's current artifact,
        recorded as an explicit `promote-artifact` attempt reading
        "skeleton checkpoint pinned", so every pointer-reading surface (the
        Working-on picker, the journey strip's rig resolution, quality
        actions, the pipeline-policy lineage) agrees with the human decision
        by construction rather than by coincidence. The one case that is not
        pinned is a run whose pointer already names an animated output: that
        artifact is downstream of the skeleton, and approving the skeleton
        must not rewind the run past its own animation.
        """
        with self._lock(run_id):
            run = self._get_unlocked(run_id)
            reviewed_rig = next((
                attempt for attempt in reversed(run.attempts)
                if attempt.stage in self._RIG_STAGES
                and attempt.status == "completed"
                and attempt.output_artifact_id == artifact_id
            ), None)
            if reviewed_rig is None:
                raise ValueError(
                    "A checkpoint requires a completed rig artifact from this run"
                )
            run.checkpoints[checkpoint] = HumanCheckpoint(note=note, artifact_id=artifact_id)
            if run.current_artifact_id != artifact_id and not self._points_at_animation(run):
                self._pin_current_unlocked(run, artifact_id, "skeleton checkpoint pinned")
            run.updated_at = time.time()
            self._save(run)
            return run

    @staticmethod
    def _points_at_animation(run: RunManifest) -> bool:
        """Is the run's current artifact an animated output of one of its own attempts?"""
        if not run.current_artifact_id:
            return False
        return any(
            attempt.stage in RunStore._ANIMATED_STAGES
            and attempt.status == "completed"
            and attempt.output_artifact_id == run.current_artifact_id
            for attempt in run.attempts
        )

    def _lock(self, run_id: str) -> FileLock:
        return FileLock(self._path(run_id).with_suffix(".lock"))

    def _get_unlocked(self, run_id: str) -> RunManifest:
        path = self._path(run_id)
        if not path.exists():
            raise KeyError(run_id)
        return self._load(path)

    def _path(self, run_id: str) -> Path:
        if not run_id or any(character not in "0123456789abcdef" for character in run_id):
            raise KeyError(run_id)
        return self.root / f"{run_id}.json"

    def _save(self, run: RunManifest) -> None:
        path = self._path(run.id)
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(asdict(run), indent=2), encoding="utf-8")
        os.replace(temporary, path)

    def _load(self, path: Path) -> RunManifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["review_status"] = ReviewState(payload["review_status"])
        # Idempotent migration: manifests written before subject_kind existed
        # default to "auto", which keeps every journey stage applicable.
        subject_kind = payload.get("subject_kind") or "auto"
        payload["subject_kind"] = subject_kind
        payload["applicable_stages"] = payload.get("applicable_stages") or applicable_stages_for(subject_kind)
        payload["attempts"] = [StageAttempt(**attempt) for attempt in payload["attempts"]]
        payload["checkpoints"] = {
            name: HumanCheckpoint(
                note=checkpoint["note"],
                artifact_id=checkpoint.get("artifact_id", payload.get("current_artifact_id") or payload["source_artifact_id"]),
                completed_at=checkpoint["completed_at"],
            )
            for name, checkpoint in payload.get("checkpoints", {}).items()
        }
        return RunManifest(**payload)
