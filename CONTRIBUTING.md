# Contributing to the AISmith engine

Thanks for helping. Bug reports, fixes, docs and tests are all welcome.

## Licence and sign-off

Contributions come in under the same licence they go out under: by
contributing, you agree that your contribution is licensed under this
repository's MIT License.

Every commit must carry a `Signed-off-by` line, which certifies the
[Developer Certificate of Origin 1.1](https://developercertificate.org/): that
you wrote the change, or otherwise have the right to submit it under the MIT
License. Add the line with `git commit -s`. It must use your real name and an
email address you can be reached at:

```
Signed-off-by: Your Name <you@example.com>
```

CI checks every commit in a pull request for the line, merge commits included,
and pull requests with unsigned commits cannot be merged. To sign commits you
have already made, run `git rebase --signoff <base-branch>` and force-push your
branch. Bring your branch up to date by rebasing rather than merging, or sign
the merge with `git merge --signoff`. GitHub's *Update branch* button makes an
unsigned merge commit.

## Before you start

- **Security problems:** do not open an issue. Follow [SECURITY.md](SECURITY.md).
- **Larger changes:** open an issue first so we can agree on the approach. This
  matters most for the stage contract and result schema
  (`backend/engine/contract.py`), the lock files, and the quality gates in
  `docs/PIPELINE_POLICY.md`.
- **Never commit** model weights, generated meshes or images, credentials,
  `studio-data/`, or art you do not have the rights to. New test images need a
  provenance record (source, licence, how they were made) beside them, like the
  existing `PROVENANCE.md` files.
- **Dependencies:** every model and third-party source is pinned in
  `vendor/weights.lock.json` or `vendor/sources.lock.json`, with its licence,
  and listed in [NOTICE](NOTICE) and
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). A new dependency under a
  copyleft, non-commercial or custom licence needs a discussion first, and an
  entry in NOTICE in the same pull request.
- **Vendored code:** fixes to `vendor/trellis2cpp` or `vendor/mesh2motion-app`
  that are not specific to this project belong upstream first.

## Set up

You need Python 3.11 or 3.12 and Node 22 or newer. Generation needs an NVIDIA
GPU, the pinned runtimes and the model weights (see
[docs/engineering/engine-runtime.md](docs/engineering/engine-runtime.md)), but
the checks below do not.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[test,meshfix]"
(cd vendor/mesh2motion-app && npm ci)
(cd frontend && npm ci)
```

## Checks

Run the checks for what you changed:

```bash
# Backend: a focused test file, then the whole CPU suite
.venv/bin/python -m pytest -q backend/tests/<affected_test>.py
.venv/bin/python -m pytest -q backend/tests
.venv/bin/ruff check backend scripts tools

# Frontend, from frontend/
npm test
npm run build
```

CI runs these, plus a secret scan, a scan for private addresses and paths, and
a licence check against NOTICE. Add a regression test for each bug you fix and
for each meaningful way your change can fail. In the pull request, say which
checks you ran and what you could not run, for example anything that needs a
GPU, Blender or the model weights.

## Boundaries

Keep these intact. A change that weakens one needs an explicit discussion first:

- **Pinned runtime.** Every source and weight is pinned by commit or hash, and
  a stage verifies its weights before it runs. Never relax a pin to a branch or
  an unhashed file.
- **Offline stages.** `aismith-engine run` and `serve` never reach the network.
- **Honest results.** A result records what actually ran (`effective_inputs`,
  the engine version and both lock digests). Never relabel a setting that was
  not applied.
- **Gates decide, scores advise.** A candidate that fails a blocking gate is
  kept but never promoted.

## Pull requests

- Keep each pull request to one change, and describe the behaviour it changes
  and why.
- Match the surrounding code's style, naming and comment density.
- Do not reformat or rename code you did not otherwise need to touch.
