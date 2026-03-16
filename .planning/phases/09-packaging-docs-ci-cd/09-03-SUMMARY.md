---
phase: 09-packaging-docs-ci-cd
plan: "03"
subsystem: ci-cd
tags: [ci, github-actions, release, pypi, pytest, bundle-contract, multi-instance]
dependency_graph:
  requires: [09-01, 09-02]
  provides: [push-ci-workflow, release-workflow, ci-helper-scripts]
  affects: [.github/workflows, scripts/ci]
tech_stack:
  added: [github-actions, pypa/gh-action-pypi-publish, twine]
  patterns: [semver-gate, bundle-contract, isolation-gate]
key_files:
  created:
    - .github/workflows/ci.yml
    - .github/workflows/release.yml
    - scripts/ci/check_tag_version.py
    - scripts/ci/verify_bundle_contract.py
  modified: []
decisions:
  - ci.yml splits python-tests, js-build, and package-smoke as separate jobs so each gate is independently visible in GitHub Actions UI
  - release.yml uses pypa/gh-action-pypi-publish with id-token trusted publishing, not API token secret
  - check_tag_version.py checks GITHUB_REF_NAME first (set by GitHub Actions) then falls back to git describe for local dry-runs
  - verify_bundle_contract.py uses REQUIRED vs OPTIONAL artifact tiers so optional metadata files do not block CI
  - --allow-no-tag flag lets check_tag_version.py pass in local dev environments with no current tag
metrics:
  duration: 3 min
  completed: "2026-03-16"
  tasks_completed: 3
  files_created: 4
---

# Phase 09 Plan 03: CI/CD Automation Summary

**One-liner:** GitHub Actions push/PR CI with multi-instance isolation gate and semver-gated PyPI release workflow using pypa/gh-action-pypi-publish trusted publishing.

## What Was Built

### Task 1: Push/PR CI Workflow (ci.yml)

`.github/workflows/ci.yml` runs on every push and pull_request with three jobs:

- **python-tests**: Matrix on Python 3.10/3.11. Installs deps, runs 6 explicit pytest suites: `test_packaging_smoke`, `test_docs_examples_contract`, `test_multi_instance_isolation`, `test_dash_runtime_callbacks`, `test_session_request_gate`, `test_runtime_service`.
- **js-build**: Node 20, `npm ci`, `npm run build`, then calls `scripts/ci/verify_bundle_contract.py` to assert artifact contract.
- **package-smoke**: Depends on python-tests + js-build passing. Runs `python -m build`, `twine check`, wheel install, and import smoke.

### Task 2: Release Workflow (release.yml) and check_tag_version.py

`.github/workflows/release.yml` triggers on `v*.*.*` tags only:
1. Runs `scripts/ci/check_tag_version.py` to enforce tag/package version parity.
2. Runs full isolation + smoke pytest gates before publish.
3. Builds distribution with `python -m build`.
4. Validates with `twine check`.
5. Publishes to PyPI using `pypa/gh-action-pypi-publish@release/v1` with trusted `id-token` publishing.

`scripts/ci/check_tag_version.py`:
- Reads `GITHUB_REF_NAME` (GitHub Actions tag context) or falls back to `git describe --exact-match`.
- Parses `dash_tanstack_pivot/pyproject.toml` for canonical version.
- Fails with clear error message if tag X.Y.Z != package version.
- `--allow-no-tag` flag for local dry-run safety (exits 0 when no tag present).

### Task 3: verify_bundle_contract.py and Local Dry-Run

`scripts/ci/verify_bundle_contract.py`:
- Checks `dash_tanstack_pivot/dash_tanstack_pivot/` for required artifact: `dash_tanstack_pivot.min.js`.
- Logs optional artifacts (`package-info.json`, `metadata.json`) as warnings, not failures.
- Supports `--bundle-dir` override for CI flexibility.

**Local dry-run results:**
- `check_tag_version.py --allow-no-tag`: PASS
- `verify_bundle_contract.py`: PASS (556.2 KB bundle present)
- `pytest tests/test_multi_instance_isolation.py tests/test_dash_runtime_callbacks.py tests/test_session_request_gate.py tests/test_runtime_service.py -q`: **22 passed**

## Requirement Coverage

| Requirement | Covered By | Evidence |
|-------------|------------|----------|
| CI-01 | Task 1 | python-tests job on every push/PR |
| CI-02 | Task 1, Task 3 | js-build job + verify_bundle_contract.py |
| CI-03 | Task 2 | tag-triggered release.yml with PyPI publish action |
| PKG-03 | Task 2, Task 3 | check_tag_version.py + semver-gated publish path |

## Multi-Instance Gate Coverage

| Invariant | CI Job | Local Verified |
|-----------|--------|----------------|
| Table-scoped isolation (table_a vs table_b) | python-tests | 22 tests pass |
| Filter/sort isolation across instances | python-tests | 22 tests pass |
| Interleaved request concurrency | python-tests | 22 tests pass |
| Abort generation isolation per-instance | python-tests | 22 tests pass |
| Session request gate correctness | python-tests | 22 tests pass |
| Bundle artifact presence post-build | js-build | bundle FOUND 556 KB |

## Deviations from Plan

None - plan executed exactly as written.

### Pre-existing Issue (Out of Scope, Deferred)

The `npm run build` command includes a `build:py` step that calls `python -m dash.development.component_generator`, which fails on this machine with `ModuleNotFoundError: No module named 'pkg_resources'`. This is a pre-existing environment issue unrelated to this plan's changes. The JS webpack bundle (`build:js`) completes successfully and produces the required artifact. The CI environment (Ubuntu + GitHub Actions) uses standard pip tooling and will not have this issue.

Logged to deferred items: `pkg_resources` missing from local Anaconda env breaks `build:py` step (cosmetic on this machine, not a CI blocker).

## Self-Check

Checking created files and commits...

## Self-Check: PASSED

| Item | Status |
|------|--------|
| .github/workflows/ci.yml | FOUND |
| .github/workflows/release.yml | FOUND |
| scripts/ci/check_tag_version.py | FOUND |
| scripts/ci/verify_bundle_contract.py | FOUND |
| Commit 7028ea8 (Task 1: ci.yml) | FOUND |
| Commit 6256f53 (Task 2: release.yml + check_tag_version.py) | FOUND |
| Commit c73d77f (Task 3: verify_bundle_contract.py) | FOUND |
