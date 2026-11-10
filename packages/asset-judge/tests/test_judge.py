import json
from copy import deepcopy
import pytest
from pydantic import ValidationError
from asset_judge import *


def identity(name="primary"):
    return Identity(provider="fixture", model=name, version="1", prompt_version="1")

def request(revision="a"):
    return Request(id="request-" + revision, artifact=Artifact(id="asset", revision=revision, sha256=revision * 64),
        scope="asset", target="Preserve the approved silhouette", rubric_version="1",
        criteria=(Criterion(id="shape", description="Faithful silhouette", threshold=.8),),
        evidence=(Evidence(id="front", uri="front.png", sha256="f"*64, artifact_sha256=revision*64,
                          producer="fixture", producer_version="1", kind="image", role="canonical", camera_id="front"),),
        canonical_cameras=("front",), conditions={"lighting": "neutral-v1"})

def result(req, execution="judge-1", score=.9, judge=None, finding=None):
    return Result(request_sha256=req.fingerprint(), judge=judge or identity(), execution_id=execution,
                  verdict="pass" if score >= .8 and finding is None else "revise", confidence=.9,
                  scores={"shape": score}, evidence_ids=("front",), rationale="Fixture evidence",
                  findings=(Finding(id=finding, criterion_id="shape", blocking=True, message="Fix " + finding,
                                    evidence_ids=("front",)),) if finding else ())

def evaluate(req=None, **kw):
    req = req or request()
    return evaluate_round(round_id=kw.pop("round_id", "1"), builder_execution_id="builder", request=req,
                          policy=kw.pop("policy", Policy(primary=identity())), primary=kw.pop("primary", result(req)),
                          hypothesis="Change one measured shape defect", **kw)

def test_roundtrip_pass_and_no_art_acceptance():
    r = evaluate()
    assert Round.model_validate_json(r.model_dump_json()) == r
    validate_history((r,))
    assert r.outcome.state == "pass" and r.outcome.human_art_acceptance is False

@pytest.mark.parametrize("mutation", [
    {"confidence": float("nan")}, {"confidence": 2}, {"verdict": "accepted"},
    {"human_art_acceptance": True}, {"scores": {"shape": float("inf")}}])
def test_result_schema_rejects(mutation):
    data = result(request()).model_dump()
    data.update(mutation)
    with pytest.raises(ValidationError):
        Result.model_validate(data)

@pytest.mark.parametrize("mutation", [
    {"request_sha256": "0"*64}, {"judge": identity("other")}, {"execution_id": "builder"},
    {"scores": {}}, {"evidence_ids": ("missing",)}, {"scores": {"shape": .1}}])
def test_invalid_result_never_passes(mutation):
    with pytest.raises(ValueError):
        evaluate(primary=result(request()).model_copy(update=mutation))

def test_missing_canonical_and_stale_evidence():
    data = request().model_dump()
    data["canonical_cameras"] = ["back"]
    with pytest.raises(ValueError): Request.model_validate(data)
    data["canonical_cameras"] = ["front"]
    data["artifact"]["sha256"] = "b"*64
    with pytest.raises(ValueError): Request.model_validate(data)

def test_disagreement_adjudication_and_original_findings_retained():
    req = request()
    policy = Policy(primary=identity(), secondary=identity("second"), adjudicator=identity("tie"))
    second = result(req, "second-exec", .4, identity("second"), "wide")
    assert evaluate(policy=policy, secondary=second).outcome.state == "needs_adjudication"
    adjudicated = evaluate(policy=policy, secondary=second, adjudication=result(req, "tie-exec", judge=identity("tie")).model_copy(update={"resolved_finding_ids": ("wide",)}))
    assert adjudicated.outcome.state == "pass"
    assert adjudicated.outcome.blocking_findings[0].id == "wide"
    assert adjudicated.outcome.human_art_acceptance is False

def test_low_confidence_abstain_and_missing_second():
    assert evaluate(primary=result(request()).model_copy(update={"confidence": .1})).outcome.state == "human_review"
    assert evaluate(primary=result(request()).model_copy(update={"verdict": "abstain"})).outcome.state == "human_review"
    assert evaluate(policy=Policy(primary=identity(), secondary=identity("second"))).outcome.state != "pass"

def series(scores, findings=None):
    history = ()
    for i, score in enumerate(scores):
        req = request("abcdef"[i])
        r = evaluate(req, round_id=str(i), primary=result(req, f"judge-{i}", score, finding=findings[i] if findings else None), history=history)
        history += (r,)
    return history

def test_plateau_oscillation_and_compaction():
    history = series([.5, .51, .5])
    assert history[-1].outcome.state == "human_review"
    assert "plateau_reassess" in history[-1].outcome.reasons
    assert len(compact_history(history, 2)) == 2
    history = series([.4, .7, .4], ["wide", "narrow", "wide"])
    assert "oscillation" in history[-1].outcome.reasons


def test_progress_and_changed_comparison_do_not_plateau():
    assert series([.3, .5, .7])[-1].outcome.state == "revise"
    h = series([.5, .5])
    req = request("c").model_copy(update={"conditions": {"lighting": "new"}})
    assert evaluate(req, round_id="3", primary=result(req, "fresh", .5), history=h).outcome.state == "revise"


def test_forged_history_reused_execution_and_policy_drift():
    h = series([.5])
    forged = h[0].model_copy(update={"outcome": Outcome(state="pass", reasons=())})
    with pytest.raises(ValueError): validate_history((forged,))
    with pytest.raises(ValueError): evaluate(round_id="2", history=h, primary=result(request(), "judge-0"))
    with pytest.raises(ValueError): evaluate(round_id="2", history=h, policy=Policy(primary=identity("new")), primary=result(request(), judge=identity("new")))


def test_camera_strategy_only_after_two_rounds():
    class Camera:
        def additional_views(self, request, round_number):
            assert round_number >= 3
            return ("rear-low",)
    req = request("c")
    policy = Policy(primary=identity())
    assert plan_cameras(req, policy, (), Camera()) == ("front",)
    h = series([.3, .5])
    assert plan_cameras(req, policy, h, Camera()) == ("front", "rear-low")
    adv = Evidence(id="rear", uri="rear.png", sha256="e"*64, artifact_sha256="c"*64,
                   producer="fixture", producer_version="1", kind="image", role="adversarial", camera_id="rear-low")
    req = req.model_copy(update={"evidence": (*req.evidence, adv)})
    with pytest.raises(ValueError): evaluate(req, primary=result(req))
    assert evaluate(req, round_id="3", primary=result(req, "fresh"), history=h).outcome.state == "pass"


def test_provider_gets_only_frozen_request_and_errors_propagate():
    class Provider:
        identity = identity()
        def evaluate(self, req):
            assert isinstance(req, Request)
            return result(req)
    r = run_round(primary=Provider(), request=request(), policy=Policy(primary=identity()),
                  round_id="1", builder_execution_id="builder", hypothesis="fixture")
    assert r.outcome.state == "pass"


def test_cli_serialized_boundary(tmp_path):
    import subprocess
    import sys
    req = request()
    payload = dict(round_id="1", builder_execution_id="builder", request=req.model_dump(),
                   policy=Policy(primary=identity()).model_dump(), primary=result(req).model_dump(), hypothesis="fixture")
    src, dest = tmp_path/"input.json", tmp_path/"round.json"
    src.write_text(json.dumps(payload))
    proc = subprocess.run([sys.executable, "-m", "asset_judge.cli", str(src), str(dest)], capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert Round.model_validate_json(dest.read_text()).outcome.state == "pass"
    assert subprocess.run([sys.executable, "-m", "asset_judge.cli", str(src), str(dest)], capture_output=True).returncode == 2


def test_command_adapter_serialization_timeout_and_malformed_result(tmp_path):
    import sys
    import subprocess
    from asset_judge.command import CommandJudge
    script = tmp_path / "provider.py"
    payload = result(request()).model_dump()
    script.write_text("import json, sys\np=json.load(sys.stdin)\nassert set(p)=={'schema_version','mode','request','request_sha256'}\nprint(" + repr(json.dumps(payload)) + ")\n")
    command = CommandJudge(identity(), (sys.executable, str(script)))
    assert command.evaluate(request()) == result(request())
    script.write_text("print('{}')")
    with pytest.raises(ValueError): command.evaluate(request())
    script.write_text("import time; time.sleep(1)")
    command = CommandJudge(identity(), (sys.executable, str(script)), timeout=.01)
    with pytest.raises(subprocess.TimeoutExpired): command.evaluate(request())


def test_budget_cannot_reset_by_changing_target_and_adjudication_requires_disposition():
    policy = Policy(primary=identity(), max_rounds=3)
    history = ()
    for i in range(3):
        req = request("abc"[i]).model_copy(update={"target": f"target-{i}"})
        history += (evaluate(req, round_id=str(i), primary=result(req, f"exec-{i}", .4), policy=policy, history=history),)
    assert "round_budget_exhausted" in history[-1].outcome.reasons
    req = request()
    policy = Policy(primary=identity(), secondary=identity("second"), adjudicator=identity("tie"))
    with pytest.raises(ValueError, match="explicitly resolve"):
        evaluate(policy=policy, secondary=result(req, "second-exec", .4, identity("second"), "wide"),
                 adjudication=result(req, "tie-exec", judge=identity("tie")))


def test_human_can_resolve_plateau_without_creating_art_acceptance():
    req = request()
    resolution = HumanResolution(request_sha256=req.fingerprint(), reviewer="Human", record_uri="decision.md",
                                 record_sha256="f"*64, decision="pass", rationale="Reference target already met",
                                 resolved_finding_ids=("wide",))
    r = evaluate(primary=result(req, score=.4, finding="wide"), human_resolution=resolution)
    assert r.outcome.state == "pass" and r.outcome.human_art_acceptance is False
    assert r.human_resolution == resolution
    validate_history((r,))
    with pytest.raises(ValueError):
        evaluate(primary=result(req, score=.4, finding="wide"), human_resolution=resolution.model_copy(update={"request_sha256":"0"*64}))


def test_repeated_disagreement_escalates_and_nonblocking_findings_remain():
    policy = Policy(primary=identity(), secondary=identity("second"))
    history = ()
    for i in range(3):
        req = request("abc"[i])
        history += (evaluate(req, round_id=str(i), policy=policy, primary=result(req, f"p-{i}"),
                            secondary=result(req, f"s-{i}", .5, identity("second")), history=history),)
    assert "repeated_disagreement" in history[-1].outcome.reasons
    assert history[-1].outcome.state == "human_review"
    req = request()
    note = Finding(id="optional", criterion_id="shape", blocking=False, message="Optional variation",
                   evidence_ids=("front",), suggested_actions=("Compare optional variation",))
    r = evaluate(primary=result(req).model_copy(update={"findings": (note,)}))
    assert r.outcome.state == "pass" and r.outcome.nonblocking_findings == (note,)


def test_published_schemas_match_runtime_contract():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "schemas"
    for model in (Request, Result, Policy, Round):
        assert json.loads((root / (model.__name__.lower() + ".schema.json")).read_text()) == model.model_json_schema()


def test_persisted_blocking_defect_triggers_reassess_without_score_plateau():
    # Scores move well beyond epsilon, so the score rule stays silent; the same
    # blocking defect id survives three distinct revisions and must escalate.
    history = series([.3, .5, .6], ["sheet", "sheet", "sheet"])
    assert "plateau_reassess" not in history[-1].outcome.reasons
    assert "defect_persisted_reassess" in history[-1].outcome.reasons
    assert history[-1].outcome.state == "human_review"
    assert "defect_persisted_reassess" not in history[-2].outcome.reasons  # window not yet full


def test_renamed_defect_needs_a_matcher_to_persist():
    renamed = ["gush-sheet", "gush-ribbon", "gush-glass"]
    plain = series([.3, .5, .6], renamed)
    assert "defect_persisted_reassess" not in plain[-1].outcome.reasons
    # A host-supplied matcher (for example a Jev same-defect probability) links the renames.
    history = ()
    for i, score in enumerate([.3, .5, .6]):
        req = request("abc"[i])
        r = evaluate(req, round_id=str(i), primary=result(req, f"judge-{i}", score, finding=renamed[i]), history=history,
                     same_defect=lambda a, b: a.id.startswith("gush") and b.id.startswith("gush"))
        history += (r,)
    assert "defect_persisted_reassess" in history[-1].outcome.reasons
    assert history[-1].outcome.state == "human_review"


def test_retry_on_same_artifact_does_not_count_as_persistence():
    req = request("e")  # one revision judged three times: retries, not attempts
    history = ()
    for i in range(3):
        history += (evaluate(req, round_id=str(i), primary=result(req, f"judge-{i}", .5, finding="sheet"), history=history),)
    assert "defect_persisted_reassess" not in history[-1].outcome.reasons
