"""Run from any directory after installing boundless-asset-judge; synthetic evidence only."""
from asset_judge import Artifact, Criterion, Evidence, Identity, Policy, Request, Result, evaluate_round

judge = Identity(provider="offline-fixture", model="fixture", version="1", prompt_version="1")
request = Request(id="example", artifact=Artifact(id="example", revision="fixture-1", sha256="a"*64),
    scope="asset", target="Fixture silhouette target", rubric_version="example-v1",
    criteria=(Criterion(id="shape", description="Match the specified silhouette", threshold=.8),),
    evidence=(Evidence(id="front", uri="fixture://front", sha256="b"*64, artifact_sha256="a"*64,
                       producer="fixture", producer_version="1", kind="image", role="canonical", camera_id="front"),),
    conditions={"lighting": "fixture-neutral", "units": "metres"}, canonical_cameras=("front",))
result = Result(request_sha256=request.fingerprint(), judge=judge, execution_id="fixture-judge-1",
                verdict="pass", confidence=.9, scores={"shape": .9}, evidence_ids=("front",),
                rationale="Synthetic protocol example; no visual evaluation occurred.")
print(evaluate_round(round_id="1", builder_execution_id="fixture-builder", request=request,
                     policy=Policy(primary=judge), primary=result, hypothesis="Synthetic example").model_dump_json(indent=2))
