# Independent asset judge

Engine-neutral Python API and versioned JSON schemas for reviewing assets, regions
and scenes. Install this package alone; AISmith's GPU/backend dependencies are not
required. The [boundary decision](DESIGN.md) explains its home and ordered gates.

```sh
python -m pip install -e 'packages/asset-judge[test]'
python -m pytest packages/asset-judge/tests
asset-judge round-input.json new-round.json
```

The CLI accepts `round_id`, `builder_execution_id`, `request`, `policy`, `primary`,
`hypothesis`, optional `secondary`, `adjudication`, `human_resolution`, and prior `history` rounds.
It writes exclusively to a new filename. Exit 0 means a model recommendation of
pass, 10 means unresolved review, 2 means invalid input. Neither exit 0 nor a
score grants human art acceptance. See [schemas](schemas) and the
[offline example](examples/offline_round.py).

## Integration API

1. Freeze an `Artifact` (revision, SHA-256, causal parent hashes), versioned rubric,
   scale/pose/light/exposure conditions and evidence inventory. Evidence references
   identify producer/version, content hash, artifact hash, type, role and camera.
   Separate structural references from aesthetic/atmospheric ones. Never send
   builder conversation, proposed self-grade, or previous judge scores to judges.
2. Pin `Policy.primary` provider/model/version/prompt-version for the entire series.
   Each round uses a fresh execution, including when the same model is used by
   the builder. Identity declarations are audit data: the host must actually
   isolate contexts. Version changes start a new explicitly retained series.
3. Submit `Result` JSON via `evaluate_round`, or implement `Judge.evaluate` and use
   `run_round`. The optional `CommandJudge` starts an explicit argv without a shell,
   passes JSON through stdin, validates output and enforces a process timeout.
   Provider wrappers attach hash-verified image/video bytes using their API's
   multimodal format, enforce provider-side deadlines, and must not spawn detached
   work. This transport is not a sandbox or a credential manager.
4. An optional second opinion uses the identical request and its own identity and
   execution. Disagreement in verdict, blocking finding IDs or scores beyond the
   configured band yields `needs_adjudication`. No averaging or judge rotation.
5. An optional adjudicator independently resolves the evidence against both
   opinions through `adjudicate(request, primary, secondary)`. Its rationale and
   `resolved_finding_ids` must explicitly cover every original blocking finding.
   Original findings remain in the outcome for audit, even after resolution.
   Low confidence or abstention escalates to `human_review`; it cannot silently pass.
6. Host records rounds immutably. Same-series history is revalidated and outcomes
   recomputed before consumption. Stable finding IDs express recurring defects,
   not per-round UUIDs. Three distinct revisions with negligible per-criterion
   progress trigger plateau reassessment (`plateau_epsilon` .05, at the measured
   same-prompt judge noise); a blocking defect that survives `persistence_window`
   distinct revisions triggers `defect_persisted_reassess` by id, or by a host
   `same_defect(a, b)` matcher for renamed findings; A/B/A blocking findings trigger
   oscillation. Three consecutive disagreements and the finite total round
   budget also escalate. Changed references/conditions reset comparability, not
   the total budget. Only substantive revisions count for plateau/oscillation.
7. `compact_history` returns bounded hypotheses, finding summaries and original
   hashes/references for the builder/human packet. Keep full JSON elsewhere; do
   not resend every old image. The judge continues to receive the current request.

`plan_cameras` always retains canonical IDs. Optional `CameraStrategy` adds views
only after at least two comparable canonical rounds; evidence submission enforces
that same minimum. The engine adapter owns pose definitions, capture constraints,
seeds and actual rendering. Baseline evidence remains required every round.
Asset/region/scene are independent scopes: a regional pass never proves every asset
or the whole scene. Hosts choose coverage; this package does not invent it.

An external human can settle an escalation using `HumanResolution`: exact request
hash, reviewer, retained decision URI/hash, pass/revise decision, rationale and
explicit dispositions of blocking finding IDs. Settlement retains the original
scores, findings and escalation reasons. The host must authenticate and verify the
human record before submission; a self-declared JSON identity is not a signature.
This resolves the review loop and still never sets `human_art_acceptance` true.
Human escalation produces a review state, not an approval record. The host retains
its technical checks and obtains/records the actual human decision or an explicit
inherited family target with current conformance evidence. No automatic publishing,
asset promotion, tracker messages or live model calls occur in the offline example.

## Validation boundary

Automated tests exercise schema/JSON/process handoffs, malformed/stale evidence,
identity/execution reuse, disagreement/disposition, confidence, iteration budgets,
plateau/oscillation and delayed cameras. These validate protocol behavior, not
visual evaluator accuracy. Qualify each real provider on retained approved/rejected
examples before treating its scores as useful; calibrate confidence and rubric
thresholds locally. No live model or renderer qualification is claimed in this slice.
