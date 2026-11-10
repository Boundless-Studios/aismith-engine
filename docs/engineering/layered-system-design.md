# Layered System Design and Validation

Use this practice for frameworks, pipelines, orchestration, cross-process
features, and any flow where several components transform or transfer state.
It prevents downstream work from hiding an upstream failure.

The core rule is:

> Sketch globally, define boundaries explicitly, validate locally, and advance
> sequentially.

Scale the ceremony to the risk. A small pure function may need only an explicit
input/output contract and a focused test. A deep flow needs the complete method
below.

## 1. Sketch the system before its internals

Start with the end-to-end shape. Name each layer in dependency order, the state
it owns, and the externally observable outcome it must produce. This sketch is
a map, not an implementation plan.

```text
source -> layer 1 -> handoff -> layer 2 -> handoff -> layer 3 -> result
```

Use this table to make each layer independently understandable and testable:

| Layer | Purpose | Inputs | Outputs | Invariants | Dependencies | Failure modes | Observability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Name | One responsibility | Typed or versioned inputs | Typed or versioned outputs | Facts always true | Required services/runtime | Expected failures | Logs, metrics, artifacts |

A layer boundary is suspect if consumers must read the producer's internals to
use it, or if the producer can change its internals only by changing consumers.

## 2. Specify every handoff as a two-sided contract

An interface is not just a function signature. Define what the producer
promises and what the consumer verifies before doing work.

| Contract field | Required definition |
| --- | --- |
| Producer and consumer | Exact components on both sides |
| Data shape | Schema, format, units, coordinate system, version, and size bounds |
| Invariants | Conditions the producer guarantees and consumer checks |
| Ownership and lifetime | Who creates, mutates, retains, and deletes the data |
| Provenance | Source IDs, hashes, versions, configuration, and causal parent |
| Concurrency | Idempotency, locking, retry, and duplicate-submission behavior |
| Cancellation and timeout | Who may stop work and how resources are released |
| Failure representation | Typed error/state and whether partial output is valid |
| Acceptance evidence | Exact observation that proves the handoff is healthy |

Test both sides. Producer tests prove valid output. Consumer tests prove invalid
input is rejected before expensive or destructive work starts. Integration
tests prove the serialized or process-level boundary, not merely two mocks that
agree with each other.

## 3. Turn the layer map into ordered health gates

For each layer and handoff, define a gate before implementation or diagnosis:

| Gate | Controlled input | Acceptance criteria | Evidence retained | Status |
| --- | --- | --- | --- | --- |
| G0 | Exact fixture/artifact | Measurable pass condition | Command, logs, IDs, output | Closed / Open |

Work from the earliest dependency forward:

```text
system sketch
  -> layer contracts
  -> handoff contracts
  -> first closed health gate
  -> one-variable experiment
  -> retained evidence
  -> open gate
  -> next layer
```

An open gate means the acceptance criteria passed with reproducible evidence.
It does not mean that the code looks plausible, a mocked unit test passed, or a
later stage happened to produce an output.

## 4. Diagnose the first failing boundary

When output is wrong or a process crashes:

1. Reproduce with the smallest controlled input that still crosses the suspect
   boundary.
2. Find the earliest boundary whose input is known-good but output is invalid,
   missing, or unproven.
3. State one falsifiable hypothesis about that boundary.
4. Change one relevant variable and rerun the same probe.
5. Record the observation and conclusion before the next experiment.
6. Fix the localized cause, rerun the boundary probe, then rerun the immediately
   adjacent integration.

Use this experiment record:

```text
Boundary:
Controlled input and content hash:
Environment/runtime/model versions:
Hypothesis:
Single changed variable:
Command or action:
Observed output, logs, runtime, and resource use:
Conclusion:
Acceptance condition and status:
Next experiment, if still closed:
```

Downstream observations may provide diagnostic clues about an earlier boundary.
They do not authorize integrated downstream validation, promotion, or a
compensating workaround while that earlier gate remains closed. Separately
approved work may implement a consumer against a versioned contract and fixture,
but it remains explicitly unvalidated in the real pipeline and must not displace
diagnosis of the earliest failing boundary.

## 5. Stop rules

Stop advancing and remain at the current boundary when:

- its input provenance is unknown;
- its acceptance criteria are missing or subjective when measurable criteria
  are available;
- the failure cannot be reproduced with retained inputs and configuration;
- an experiment changes several variables at once;
- a downstream workaround would accept malformed or unverified upstream data;
- cleanup, timeout, cancellation, or ownership behavior is unproven;
- the result exists but does not satisfy the capability being claimed.

Do not redesign an entire framework because one deep-flow run failed. Redesign
only after boundary evidence shows that the existing abstraction or contract is
the cause and a local correction cannot restore its invariant.

## 6. Evidence and promotion

Retain enough information for another contributor to repeat the result:

- immutable input IDs or hashes;
- code revision and branch;
- runtime, dependency, driver, model, and workflow versions;
- expanded configuration with secrets redacted;
- exact command or user action;
- timestamps, duration, memory/VRAM, logs, and terminal state;
- output IDs or hashes and measurable quality results;
- visual comparisons when quality is visual;
- the human or automated decision that opened or kept closed the gate.

An artifact may be retained for diagnosis without being promoted. Promotion is
a separate decision and requires its own explicit acceptance gate.

## 7. Goal mode and long-running agents

Goal mode changes persistence, not engineering standards. An autonomous agent
must continue at the earliest unvalidated boundary until it passes or a genuine
external blocker is established. It must not skip forward simply because later
work is available.

At every durable handoff or context rotation, record:

```text
Goal:
System sketch/version:
Last open gate:
Earliest unvalidated boundary:
Confirmed evidence:
Current falsifiable hypothesis:
Next single experiment:
Exact acceptance condition:
Blocked dependencies, if any:
Explicitly closed downstream work:
```

“Continue,” “do not stop,” or a large remaining budget authorizes persistence
within scope. It does not authorize broader rewrites, unreviewed external state
changes, or advancing past a closed gate.

## 8. AISmith example

The character pipeline should be validated in this order:

| Gate | Boundary | Example acceptance evidence |
| --- | --- | --- |
| G0 | Prepared-image approval -> run state | Promoted artifact ID and content hash exactly match the reviewed image |
| G1 | Run state -> geometry request | Request records the same source artifact, profile, and execution provenance |
| G2 | API/coordinator -> worker startup | Correct executable, models, device, arguments, ownership, and cancellation behavior |
| G3 | Worker -> image preprocessing | Dimensions, channels, alpha/background treatment, normalization, and tensor shape are verified |
| G4 | Preprocessing -> encoder/sampler | The controlled input completes without compiler/runtime corruption and records runtime/VRAM |
| G5 | Sampler -> geometry extraction | A valid mesh artifact is produced with deterministic provenance |
| G6 | Geometry -> quality evaluation | Silhouette, holes, connected components, limb separation, topology, texture fidelity, runtime, and VRAM meet explicit thresholds |
| G7 | Accepted geometry -> retopology | Retopology preserves accepted shape and closes only its own topology criteria |
| G8 | Retopology -> skeleton placement | Skeleton landmarks and bind pose are valid for the accepted mesh |
| G9 | Skeleton -> weights | Weight coverage, normalization, locality, and deformation readiness pass |
| G10 | Weights -> deformation/animation | Review poses and target animation meet explicit visual and numeric criteria |

If the compiler crashes at G4, G5 through G10 remain closed. Investigate the
exact G3-to-G4 tensor/runtime handoff first. Do not tune mesh cleanup, retopology,
weights, or animation to compensate. If G4 passes but G6 fails, diagnose geometry
generation and extraction before touching rigging.

This ordering makes failures smaller, preserves causal evidence, and prevents a
later layer from disguising an earlier defect.
