# Astra as pipeline operator

`backend/mcp_server.py` is a stdio MCP server that lets an external agent —
OpenAI GPT-6 Astra via its Responses API `mcp` tool, or Claude Code, or any
other MCP client — drive the AISmith pipeline the same way a human operator
would: through the HTTP API in `backend/app/main.py`, one tool call at a
time. This document is the delegation contract: what moves out to the
agent, what stays in the deterministic system, and the guardrails that keep
the boundary where PIPELINE_POLICY.md drew it.

## Purpose

The pipeline (`docs/PIPELINE_POLICY.md`, S0 through S6) is deterministic and
contract-checked by design. It does not need an agent to run it — it needs
one to make the handful of decisions it deliberately refuses to make for
itself: which source image, which prompt and seed to spend GPU time on,
which motion clip a character performs, whether a finished asset is
accepted. An agent acting as the **operator** supplies those decisions and
nothing else. It never edits an artifact, never reaches around the API into
the filesystem, and every decision it makes is recorded on the run the same
way a human's would be. The pipeline stays the source of truth; the agent
is a second pair of hands with an explicit, narrow mandate.

## What moves out to the agent

Everything the pipeline itself is built to refuse (see the "Deliberately
absent" note above `STAGE_JOB_KINDS` in `backend/app/pipeline_policy.py`):

- **`STAGE_MANUAL_INPUT`** — stages whose next step is a human action, not a
  job:
  - `S0_SOURCE` → `"source-image"`: which reference image starts the run.
  - `S1_GENERATION` → `"generation-request"`: the prompt/seed/detail-level
    choice that spends GPU on TRELLIS.2 geometry generation.
  - `S6_EXPORT` → `"operator-accept"`: the final accept gate — by
    definition a human decision; there is no job to submit.
- **`STAGE_CREATIVE_INPUT`** — a stage with a real job kind whose job is
  meaningless without a choice named first:
  - `S5_ANIMATION` → `"clip"`: which motion clip the character performs.
    The driver may submit `apply-clip`, but only once a caller has named
    the clip.
- **Post-cap remediation decisions.** `DEFAULT_STAGE_ATTEMPT_CAP` (2) bounds
  how many times the loop will re-run the same stage as remediation before
  it stops asking the GPU and asks a human instead. Once a stage hits the
  cap, deciding what happens next — accept the defect, change the input,
  try a different seed — is the agent's call.
- **Rig-adjust joint edits.** Fixing a placed joint by hand happens in
  Blender on the exported GLB, outside this pipeline entirely; the edited
  GLB comes back in via `submit_job(kind="rig-adjust")`, and the agent is
  the one deciding an adjustment is needed and supplying the result.

## What stays in our system

Everything else is the pipeline's, not the agent's, to run or to judge:

- The deterministic loop itself — `next_action`, one step per call, never
  chained.
- Contracts and the blame table — which stage owns which defect.
- Every measurement gate — an agent cannot talk a contract into passing.
- Artifact lineage and immutability — artifacts are never edited in place.
- GPU job execution — the worker processes, not the agent, do the work.
- Export contracts — S6's shape is fixed by the pipeline, not negotiated.

`docs/PIPELINE_POLICY.md`'s principles say this directly:

> 1. **Contracts, not repairs.** Every stage declares measurable invariants on its output. A
>    stage may only consume inputs that pass the previous stage's contract. "Repair" steps that
>    compensate downstream for upstream violations are forbidden as permanent fixtures — they
>    may exist only as temporary guards, each with a filed remediation for the owning stage.
>
> 3. **Determinism.** Same inputs + same seed + same stage versions ⇒ same artifact. Every
>    artifact records provenance (inputs, settings, seed, stage impl version). Re-running the
>    loop after a remediation is mechanical: invalidate from the remediated stage, re-run
>    downstream, re-evaluate contracts.
>
> 5. **Human judgment is a stage, not a patch.** Operator decisions (identity/likeness, final
>    accept) are explicit gates with recorded outcomes; they never modify artifacts.

Principle 5 is exactly this document's mandate, generalized from "human" to
"operator" — an agent supplying those decisions is filling the same
recorded-gate role a person does, not a new kind of authority.

## Tool-call sequence for one full run

1. `health()` — confirm the API is reachable before doing anything else.
2. `list_runs()` / `get_run(run_id)` — find or inspect the run.
3. `get_run_policy(run_id)` — read the whole-run evaluation: per-stage
   status, `next_action`, blame.
4. Loop on `pipeline_step(run_id, settings)`:
   - `advanced` / `remediated` — a job was submitted; move to step 5.
   - `needs_input` — supply the named setting (e.g. `{"clip": "Wave"}`) and
     call again, or go to step 6 if the input is a spend decision that
     belongs in `submit_job` instead (`source-image`,
     `generation-request`).
   - `capped` — the stage has hit `DEFAULT_STAGE_ATTEMPT_CAP`; decide
     whether to accept, adjust, or escalate. Do not raise the cap without
     `operator_confirmed=True`.
   - `done` — S6 is green; go to step 8.
5. For a spend decision (`generate`, `create-tpose-set`,
   `synthesize-views`, `prepare-image`, or a Blender-side `rig-adjust`):
   `submit_job(kind, source_artifact_id, run_id, settings,
   operator_confirmed=True)`.
6. `job_status(job_id)` — poll until the job leaves `queued`/`running`.
7. `diagnose_rig(artifact_id)` for rig health, or `download_artifact` to
   pull the asset for visual inspection, as needed before deciding whether
   to proceed.
8. `record_checkpoint(run_id, "skeleton", note, artifact_id)` when a rig
   checkpoint needs the operator's recorded approval.
9. `promote_artifact(run_id, artifact_id)` to make an artifact the run's
   current one.
10. Repeat from step 4 until `done`, then the S6 operator-accept checkpoint
    closes the run.

## Guardrails

| Tool | Refusal condition | What the agent does instead |
| --- | --- | --- |
| `pipeline_step` | `next_action`'s stage is in `STAGE_MANUAL_INPUT` or `STAGE_CREATIVE_INPUT` and the named input is missing from `settings` | Supply the input in `settings` (e.g. `{"clip": "..."}`), or handle the decision through `submit_job` / `record_checkpoint` if it is not a `pipeline_step` input at all |
| `pipeline_step` | Caller tries to raise `attempt_cap` above the default without confirming | Pass `operator_confirmed=True` if deliberately widening the remediation bound; otherwise accept the default cap |
| `submit_job` | `kind` is `generate`, `create-tpose-set`, `synthesize-views`, or `prepare-image` and `operator_confirmed` is not `True` | Set `operator_confirmed=True` once the spend decision is deliberate |
| `download_artifact` | `dest_path` is not an absolute path under `AISMITH_MCP_DOWNLOAD_DIR` | Pass an absolute path under the configured download directory |

None of the guardrails are enforced only client-side for comfort: the
server's own `pipeline/step` endpoint independently returns `needs_input` or
`capped` for the same conditions (`backend/app/main.py`'s
`pipeline_step`), and `pipeline_policy.STAGE_MANUAL_INPUT` /
`STAGE_CREATIVE_INPUT` / `DEFAULT_STAGE_ATTEMPT_CAP` are the same tables
both sides read — the MCP server just refuses locally first so a caller
never spends a round trip to be told no.

## Attaching from Claude Code

Add to `.mcp.json`:

```json
{
  "mcpServers": {
    "aismith": {
      "command": "<repo>/.venv/bin/python",
      "args": ["-m", "backend.mcp_server"],
      "env": {
        "AISMITH_API_URL": "http://127.0.0.1:8000"
      }
    }
  }
}
```

Astra's Responses API `mcp` tool needs the server reachable over HTTP/SSE
rather than stdio launched by a local client — standing up that transport
(or a stdio-to-HTTP bridge in front of this server) is out of scope here.
