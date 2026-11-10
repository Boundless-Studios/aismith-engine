# Independent artifact evaluation: incremental boundary

AISmith owns asset diagnostics, run lineage and promotion. The neighbouring
tools (character intake, code-review settlement, leases, session lifetime) are
not visual judges. Keep this separately installable package in AISmith, without
importing its GPU, FastAPI, Blender or rigging stack. The consuming game project
owns rubrics and maturity, never the generic verdict algorithm.

Ordered layers and two-sided handoffs:

| Layer | Producer promise / consumer check | Output and health gate |
| --- | --- | --- |
| Evidence | Host freezes source/export hashes, references, scope, conditions and producer versions; schema rejects absent lineage and ambiguous IDs | Immutable request fingerprint; roundtrip and stale-result tests |
| Execution | Host selects a stable primary identity and fresh execution; evaluator receives only request, no builder conversation or prior scores | Versioned result tied to exact request; identity and independence checks |
| Settlement | Core validates all results before use; optional second opinion is for the same evidence | Disagreement cannot be averaged away; explicit adjudicator or human escalation |
| Iteration | Host retains immutable rounds; core compares only same target/rubric/conditions/policy | Score plateau, defect persistence (by id or host matcher), A/B/A oscillation, finite budget, compact failure history |
| Camera | Host supplies canonical captures; strategy proposes additional IDs after two baseline rounds | Canonical views remain mandatory; engine host owns actual camera poses/captures |
| Maturity | The consuming project weighs the recommendation beside measured gates and a recorded taste decision | Missing/stale evidence cannot promote; scores never create human acceptance |

Requests/results are JSON, SHA-256 content-bound, version 1. Scores/confidence
are finite 0..1, scope is asset/region/scene. Evidence units and coordinates are
host-owned explicit condition strings. All results are untrusted data; suggested
actions are recommendations, never commands. Host runs producers and judges with
bounded time/resources and retains their failures. Provider adapters must start
fresh sessions and attach actual evidence bytes/URLs after hash verification;
identity metadata alone cannot prove process isolation.

The Python API is pure and has no shared mutable ledger. CLI inputs are immutable
snapshots; outputs use exclusive creation (no overwrite). One orchestrator owns
append/order and durable storage. Multiple writers must use their own transactional
store; this package does not claim distributed locking. Repeating a round ID is
rejected. Cancellation/provider exceptions propagate before any round is returned;
no partial result counts as a pass. No model/service is selected or billed here.
