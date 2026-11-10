# Engine releases

This repository is the 3D engine that Magic Studios drives through its hosted
GPU worker pool,
[outbound-gpu-worker-pool](https://github.com/Boundless-Studios/outbound-gpu-worker-pool).
The pool never follows a branch. Its machine profile pins one engine release:
a tag, the tag's commit, the sha256 of both lock files, and the hash of every
model weight that release expects. Engine changes reach Studio users only
through a new tag, a pool profile that pins it, an updated worker image and a
Studio pin bump.

## Checks

CI (`.github/workflows/ci.yml`) runs on GitHub-hosted runners on every pull
request and on the default branch:

- **Backend (CPU).** `pip install -e ".[test,meshfix]"`, `ruff check` (syntax
  errors and undefined names; see `[tool.ruff.lint]` in `pyproject.toml`) and
  the full `backend/tests` suite, including the lock tests
  (`test_engine_sources.py`, `test_engine_weights.py`). No GPU, model weights,
  Blender or ComfyUI is needed; tests that would need them stub the runtime or
  skip.
- **Frontend.** `npm ci` in `vendor/mesh2motion-app` and `frontend`, then
  `npm test` and `npm run build`.

GPU qualification needs weights and a GPU, so it runs outside CI: the pool
qualifies each release against its machine profile, and the release notes
carry the result.

## Cutting a release

1. Set `version` in `pyproject.toml` (semantic versioning; the stage contract
   and the result schema count as the public interface).
2. Merge to the default branch with green CI.
3. Tag the merge commit `engine-v<version>` and push the tag. CI refuses a
   tag that does not match `pyproject.toml`.
4. Record the commit and the sha256 of `vendor/sources.lock.json` and
   `vendor/weights.lock.json` in the release notes; the pool pins all three.

A tag is never moved or reused. A fix is a new version.
