"""Pure settlement. No provider calls, project stages, or acceptance mutations."""
from __future__ import annotations
from typing import Callable, Protocol
from .models import Finding, HumanResolution, Identity, Outcome, Policy, Request, Result, Round

SameDefect = Callable[[Finding, Finding], bool]


def _persisted_defects(candidates: list[Result], same_defect: SameDefect | None) -> list[Finding]:
    """Blocking findings of the newest result that appear, same criterion, in every
    earlier candidate: by identical id, or by the host's matcher for renamed ids."""
    current, earlier = candidates[-1], candidates[:-1]
    persisted = []
    for f in current.findings:
        if not f.blocking:
            continue
        def present(r: Result):
            for g in r.findings:
                if not g.blocking or g.criterion_id != f.criterion_id:
                    continue
                if g.id == f.id or (same_defect is not None and same_defect(g, f)):
                    return True
            return False
        if all(present(r) for r in earlier):
            persisted.append(f)
    return persisted


def validate_result(request: Request, result: Result, identity: Identity):
    if result.request_sha256 != request.fingerprint() or result.judge != identity:
        raise ValueError("stale request or unexpected judge identity/version")
    criteria = {c.id: c for c in request.criteria}
    evidence = {e.id for e in request.evidence}
    if set(result.scores) != set(criteria):
        raise ValueError("scores must cover exactly the rubric")
    if not set(result.evidence_ids) <= evidence:
        raise ValueError("unknown evidence reference")
    if len({f.id for f in result.findings}) != len(result.findings):
        raise ValueError("duplicate finding ID")
    for f in result.findings:
        if f.criterion_id not in criteria or not set(f.evidence_ids) <= evidence:
            raise ValueError("finding refers to unknown criterion/evidence")
    if result.verdict == "pass" and (any(f.blocking for f in result.findings) or any(
        c.blocking and result.scores[c.id] < c.threshold for c in request.criteria
    )):
        raise ValueError("pass contradicts blocking finding or required score")


def _disagree(a: Result, b: Result, policy: Policy):
    return (a.verdict != b.verdict or
            {(f.criterion_id, f.id) for f in a.findings if f.blocking} !=
            {(f.criterion_id, f.id) for f in b.findings if f.blocking} or
            any(abs(a.scores[k] - b.scores[k]) > policy.score_disagreement for k in a.scores))


def _settle(request, policy, primary, secondary, adjudication):
    results = [primary] + ([secondary] if secondary else [])
    reasons = []
    state = "pass" if primary.verdict == "pass" else "revise"
    if policy.secondary and not secondary:
        state, reasons = "needs_adjudication", ["missing_second_opinion"]
    if secondary and _disagree(primary, secondary, policy):
        if adjudication:
            needed = {f.id for r in (primary, secondary) for f in r.findings if f.blocking}
            if not needed <= set(adjudication.resolved_finding_ids):
                raise ValueError("adjudication must explicitly resolve every disputed blocking finding")
            results.append(adjudication)
            state = "pass" if adjudication.verdict == "pass" else "revise"
            reasons.append("adjudicated_disagreement")
        else:
            state, reasons = "needs_adjudication", ["judge_disagreement"]
    elif adjudication:
        raise ValueError("adjudication requires an actual disagreement")
    if any(r.verdict == "abstain" or r.confidence < policy.min_confidence for r in results):
        state, reasons = "human_review", reasons + ["insufficient_judge_confidence"]
    # All original findings remain visible, including explicitly adjudicated ones.
    findings = tuple(f for r in results for f in r.findings)
    return Outcome(state=state, reasons=tuple(reasons),
                   blocking_findings=tuple(f for f in findings if f.blocking),
                   nonblocking_findings=tuple(f for f in findings if not f.blocking),
                   suggested_actions=tuple(dict.fromkeys(a for r in results for a in r.suggested_actions)))


def _validate_round_inputs(request, policy, primary, secondary, adjudication, builder, used):
    for result, identity in ((primary, policy.primary), (secondary, policy.secondary),
                             (adjudication, policy.adjudicator)):
        if result is None:
            continue
        if identity is None:
            raise ValueError("unconfigured judge slot")
        validate_result(request, result, identity)
        if result.execution_id == builder or result.execution_id in used:
            raise ValueError("judge must use a fresh independent execution")
        used.add(result.execution_id)


def validate_history(history: tuple[Round, ...]):
    """Recompute rather than trusting serialized outcomes supplied by the host."""
    verified = []
    for old in history:
        recomputed = _evaluate(old.id, old.builder_execution_id, old.request, old.policy,
                               old.primary, old.secondary, old.adjudication,
                               old.hypothesis, tuple(verified), old.human_resolution)
        if recomputed != old:
            raise ValueError("history contains a forged/inconsistent outcome")
        verified.append(recomputed)


def evaluate_round(*, round_id: str, builder_execution_id: str, request: Request,
                   policy: Policy, primary: Result, hypothesis: str,
                   secondary: Result | None = None, adjudication: Result | None = None,
                   history: tuple[Round, ...] = (), human_resolution: HumanResolution | None = None,
                   same_defect: SameDefect | None = None) -> Round:
    # Revalidate even caller-constructed/model_copy values and nested mutable containers.
    request = Request.model_validate(request.model_dump())
    policy = Policy.model_validate(policy.model_dump())
    primary = Result.model_validate(primary.model_dump())
    secondary = Result.model_validate(secondary.model_dump()) if secondary else None
    adjudication = Result.model_validate(adjudication.model_dump()) if adjudication else None
    human_resolution = HumanResolution.model_validate(human_resolution.model_dump()) if human_resolution else None
    history = tuple(Round.model_validate(r.model_dump()) for r in history)
    validate_history(history)
    return _evaluate(round_id, builder_execution_id, request, policy, primary,
                     secondary, adjudication, hypothesis, history, human_resolution, same_defect)


def _evaluate(round_id, builder, request, policy, primary, secondary, adjudication, hypothesis, history, human_resolution=None, same_defect=None):
    if round_id in {r.id for r in history}:
        raise ValueError("round ID already exists")
    if history and any(r.policy != policy for r in history):
        raise ValueError("judge/policy change requires a new series")
    used = {r.builder_execution_id for r in history}
    used.update(x.execution_id for r in history for x in (r.primary, r.secondary, r.adjudication) if x)
    if builder in {x.execution_id for r in history for x in (r.primary, r.secondary, r.adjudication) if x}:
        raise ValueError("builder cannot reuse a judge execution")
    _validate_round_inputs(request, policy, primary, secondary, adjudication, builder, used)
    comparable = [r for r in history if r.request.comparison_key() == request.comparison_key()]
    # Changed targets/lighting reset the local window, but never the total budget.
    if any(e.role == "adversarial" for e in request.evidence) and len([
        r for r in comparable if r.request.canonical_cameras
    ]) < policy.adversarial_after:
        raise ValueError("adversarial evaluation requires baseline canonical rounds")
    outcome = _settle(request, policy, primary, secondary, adjudication)
    reasons = list(outcome.reasons)
    if outcome.state != "pass":
        recent = comparable[-(policy.plateau_window - 1):]
        candidates = [r.primary for r in recent] + [primary]
        # Repeated scores for the identical artifact are retries, not substantive attempts.
        distinct = {r.request.artifact.sha256 for r in recent} | {request.artifact.sha256}
        if len(candidates) == policy.plateau_window and len(distinct) == policy.plateau_window:
            blocking = [c.id for c in request.criteria if c.blocking]
            if blocking and all(max(r.scores[k] for r in candidates) - min(r.scores[k] for r in candidates)
                                <= policy.plateau_epsilon for k in blocking):
                reasons.append("plateau_reassess")
            signatures = [tuple(sorted((f.criterion_id, f.id) for f in r.findings if f.blocking)) for r in candidates]
            if len(signatures) >= 3 and signatures[-1] == signatures[-3] != signatures[-2]:
                reasons.append("oscillation")
        # Defect persistence: the score rule sits near judge noise, so a blocking defect
        # that survives `persistence_window` substantive attempts is the reliable signal.
        # Distinct artifacts only, as above: re-judging the same bytes is a retry.
        recent_p = comparable[-(policy.persistence_window - 1):]
        candidates_p = [r.primary for r in recent_p] + [primary]
        distinct_p = {r.request.artifact.sha256 for r in recent_p} | {request.artifact.sha256}
        if len(candidates_p) == policy.persistence_window and len(distinct_p) == policy.persistence_window:
            if _persisted_defects(candidates_p, same_defect):
                reasons.append("defect_persisted_reassess")
        if len(history) + 1 >= policy.max_rounds:
            reasons.append("round_budget_exhausted")
        if "judge_disagreement" in reasons and len(comparable) >= 2 and all(
            "judge_disagreement" in r.outcome.reasons for r in comparable[-2:]
        ):
            reasons.append("repeated_disagreement")
    if any(r in reasons for r in ("plateau_reassess", "defect_persisted_reassess", "oscillation", "round_budget_exhausted", "repeated_disagreement")):
        outcome = outcome.model_copy(update={"state": "human_review"})
    if human_resolution:
        if human_resolution.request_sha256 != request.fingerprint():
            raise ValueError("human resolution is for a different request")
        if not {f.id for f in outcome.blocking_findings} <= set(human_resolution.resolved_finding_ids):
            raise ValueError("human resolution must dispose of blocking findings")
        reasons.append("human_adjudicated")
        outcome = outcome.model_copy(update={"state": human_resolution.decision})
    outcome = outcome.model_copy(update={"reasons": tuple(reasons)})
    return Round(id=round_id, builder_execution_id=builder, request=request, policy=policy,
                 primary=primary, secondary=secondary, adjudication=adjudication,
                 hypothesis=hypothesis, human_resolution=human_resolution, outcome=outcome)


class Adjudicator(Protocol):
    identity: Identity
    def adjudicate(self, request: Request, primary: Result, secondary: Result) -> Result:
        """Resolve conflicting findings explicitly using the frozen evidence."""


class Judge(Protocol):
    identity: Identity
    def evaluate(self, request: Request) -> Result:
        """Fresh isolated execution, exact target/evidence only; enforce timeout in adapter."""


def run_round(*, primary: Judge, secondary: Judge | None = None,
              adjudicator: Adjudicator | None = None, **kwargs) -> Round:
    """Provider-neutral orchestration; no prior scores/history reach judge callbacks."""
    request, policy = kwargs["request"], kwargs["policy"]
    validate_history(kwargs.get("history", ()))
    if kwargs.get("history") and any(r.policy != policy for r in kwargs["history"]):
        raise ValueError("judge/policy change requires a new series")
    if policy.secondary and secondary is None:
        raise ValueError("configured second opinion adapter is required")
    for adapter, expected in ((primary, policy.primary), (secondary, policy.secondary),
                              (adjudicator, policy.adjudicator)):
        if adapter and adapter.identity != expected:
            raise ValueError("adapter identity does not match policy")
    first = primary.evaluate(Request.model_validate_json(request.model_dump_json()))
    second = secondary.evaluate(Request.model_validate_json(request.model_dump_json())) if secondary else None
    initial = evaluate_round(primary=first, secondary=second, **kwargs)
    if second and _disagree(first, second, policy) and adjudicator:
        # Dedicated adjudication call receives both opinions, never builder conversation.
        tie = adjudicator.adjudicate(Request.model_validate_json(request.model_dump_json()), first, second)
        return evaluate_round(primary=first, secondary=second, adjudication=tie, **kwargs)
    return initial


def compact_history(history: tuple[Round, ...], limit: int = 3):
    if limit < 1:
        raise ValueError("limit must be positive")
    validate_history(history)
    return [{"round": r.id, "artifact": r.request.artifact.sha256,
             "request": r.request.fingerprint(), "hypothesis": r.hypothesis,
             "state": r.outcome.state, "reasons": r.outcome.reasons,
             "findings": [{"id": f.id, "message": f.message, "evidence": f.evidence_ids}
                          for f in r.outcome.blocking_findings]} for r in history[-limit:]]


class CameraStrategy(Protocol):
    def additional_views(self, request: Request, round_number: int) -> tuple[str, ...]: ...


def plan_cameras(request: Request, policy: Policy, history: tuple[Round, ...],
                 strategy: CameraStrategy | None = None) -> tuple[str, ...]:
    validate_history(history)
    baseline = [r for r in history if r.policy == policy and
                r.request.comparison_key() == request.comparison_key() and r.request.canonical_cameras]
    extra = strategy.additional_views(request, len(baseline) + 1) if strategy and len(baseline) >= policy.adversarial_after else ()
    return tuple(dict.fromkeys((*request.canonical_cameras, *extra)))
