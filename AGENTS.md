# AGENTS.md

AISmith 3D is a local image-to-3D, refinement, rigging, and animation pipeline.
Repository commands and architecture are documented in `README.md`.

## Deep-flow engineering contract

For frameworks, pipelines, cross-process features, and other deep multi-step
flows, read and follow
[`docs/engineering/layered-system-design.md`](docs/engineering/layered-system-design.md).
Its gates apply to design, implementation, debugging, and autonomous execution.

The non-negotiable rules are:

1. Sketch the broad system and its ordered layers before changing components.
2. Define every interface and handoff from both the producer and consumer side.
3. Validate layers in dependency order and stop at the first failing or
   unproven boundary.
4. Do not patch a later layer to compensate for an unhealthy earlier layer.
5. Run one-variable diagnostic experiments and retain reproducible evidence.
6. Do not redesign a framework until evidence localizes the failure to that
   framework or one of its contracts.
7. Open the next layer only after the current layer's acceptance gate passes.

Goal mode, autonomous loops, and instructions to continue do not relax these
rules. Persistence means continuing to investigate the earliest unvalidated
boundary. It does not authorize skipping gates, broadening scope, or treating
activity as progress. Before a context rotation or handoff, record the earliest
unvalidated boundary, confirmed evidence, current hypothesis, next single
experiment, and exact acceptance condition.

Apply the practice proportionally. A local pure-function change needs a concise
contract and focused test; a multi-stage or cross-process system needs the full
layer map, handoff contracts, health gates, and evidence record.
