"""Versioned JSON contracts; unknown fields and non-finite scores fail closed."""
from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Text = Annotated[str, Field(min_length=1)]
Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Unit = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]

class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

class Identity(Model):
    provider: Text
    model: Text
    version: Text
    prompt_version: Text

class Artifact(Model):
    id: Text
    revision: Text
    sha256: Hash
    parents: tuple[Hash, ...] = ()

class Evidence(Model):
    id: Text
    uri: Text
    sha256: Hash
    artifact_sha256: Hash
    producer: Text
    producer_version: Text
    kind: Literal["image", "video", "report", "source", "reference"]
    role: Literal["canonical", "adversarial", "structural", "aesthetic", "technical"]
    camera_id: str | None = None

class Criterion(Model):
    id: Text
    description: Text
    threshold: Unit
    blocking: bool = True

class Request(Model):
    schema_version: Literal[1] = 1
    id: Text
    artifact: Artifact
    scope: Literal["asset", "region", "scene"]
    target: Text
    rubric_version: Text
    criteria: tuple[Criterion, ...] = Field(min_length=1)
    evidence: tuple[Evidence, ...] = Field(min_length=1)
    conditions: dict[str, Text] = Field(min_length=1)
    canonical_cameras: tuple[Text, ...] = ()

    @model_validator(mode="after")
    def consistent(self):
        for values in ([c.id for c in self.criteria], [e.id for e in self.evidence], self.canonical_cameras):
            if len(set(values)) != len(values):
                raise ValueError("duplicate criterion, evidence or camera ID")
        for e in self.evidence:
            if e.kind != "reference" and e.artifact_sha256 != self.artifact.sha256:
                raise ValueError("evidence is for a different artifact")
            if e.role in ("canonical", "adversarial") and (not e.camera_id or e.kind not in ("image", "video")):
                raise ValueError("camera evidence needs an image/video and camera_id")
        seen = {e.camera_id for e in self.evidence if e.role == "canonical"}
        if not set(self.canonical_cameras) <= seen:
            raise ValueError("missing canonical capture")
        return self

    def fingerprint(self):
        return digest(self.model_dump(mode="json"))

    def comparison_key(self):
        return digest({"artifact_id": self.artifact.id, "scope": self.scope,
                       "target": self.target, "rubric": self.rubric_version,
                       "criteria": [c.model_dump() for c in self.criteria],
                       "conditions": self.conditions, "cameras": self.canonical_cameras,
                       "references": [e.model_dump() for e in self.evidence if e.kind == "reference"]})

class Finding(Model):
    id: Text
    criterion_id: Text
    blocking: bool
    message: Text
    evidence_ids: tuple[Text, ...] = Field(min_length=1)
    suggested_actions: tuple[Text, ...] = ()

class Result(Model):
    schema_version: Literal[1] = 1
    request_sha256: Hash
    judge: Identity
    execution_id: Text
    verdict: Literal["pass", "revise", "abstain"]
    confidence: Unit
    scores: dict[str, Unit]
    findings: tuple[Finding, ...] = ()
    evidence_ids: tuple[Text, ...] = Field(min_length=1)
    suggested_actions: tuple[Text, ...] = ()
    rationale: Text
    resolved_finding_ids: tuple[Text, ...] = ()

class Policy(Model):
    primary: Identity
    secondary: Identity | None = None
    adjudicator: Identity | None = None
    min_confidence: Unit = .7
    score_disagreement: Unit = .2
    plateau_window: int = Field(default=3, ge=3)
    # Same-prompt judge noise on identical images measured .01-.05 per criterion
    # (CINDER brine series, 2026-09-16); an epsilon below that could never fire.
    plateau_epsilon: Unit = .05
    # A blocking defect that survives this many substantive rounds triggers
    # reassessment regardless of score movement. Judges are told to reuse finding
    # ids; hosts may supply a `same_defect` matcher for renamed findings.
    persistence_window: int = Field(default=3, ge=2)
    max_rounds: int = Field(default=8, ge=3)
    adversarial_after: int = Field(default=2, ge=2)

    @model_validator(mode="after")
    def independent_slots(self):
        identities = [i for i in (self.primary, self.secondary, self.adjudicator) if i]
        if len({digest(i.model_dump()) for i in identities}) != len(identities):
            raise ValueError("opinion/adjudication slots must have distinct identities")
        return self

class Outcome(Model):
    state: Literal["pass", "revise", "needs_adjudication", "human_review"]
    reasons: tuple[str, ...]
    blocking_findings: tuple[Finding, ...] = ()
    nonblocking_findings: tuple[Finding, ...] = ()
    suggested_actions: tuple[str, ...] = ()
    human_art_acceptance: Literal[False] = False

class HumanResolution(Model):
    request_sha256: Hash
    reviewer: Text
    record_uri: Text
    record_sha256: Hash
    decision: Literal["pass", "revise"]
    rationale: Text
    resolved_finding_ids: tuple[Text, ...] = ()


class Round(Model):
    id: Text
    builder_execution_id: Text
    request: Request
    policy: Policy
    primary: Result
    secondary: Result | None = None
    adjudication: Result | None = None
    hypothesis: Text
    human_resolution: HumanResolution | None = None
    outcome: Outcome


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
