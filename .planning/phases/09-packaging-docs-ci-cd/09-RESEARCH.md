# Phase 9: Packaging, Docs & CI/CD - Research

**Researched:** 2026-03-15
**Domain:** Python packaging, Dash component distribution, documentation strategy, GitHub Actions CI/CD, PyPI publishing, multi-instance safety
**Confidence:** HIGH

## Summary

Phase 9 is mostly an operational hardening phase, but there are several structural blockers that must be planned explicitly:

1. Packaging metadata is inconsistent across the repo (`pyproject.toml` at root and `pivot_engine/pyproject.toml` both describe `scalable-pivot-engine`; `dash_tanstack_pivot` uses `setup.py` and has no runtime deps).
2. The Dash component package currently has a broken publish hook (`prepublishOnly` calls missing `_validate_init.py`).
3. There is no root `README.md`, no `CHANGELOG.md`, and no `.github/workflows/`.
4. Multi-instance runtime behavior already exists in code and tests, but docs and packaging do not surface it as a first-class contract.

The most important planning decision is to define one canonical distribution strategy for `dash-tanstack-pivot` and make versioning, CI, and docs follow that strategy. The plan should treat multi-instance safety as a release gate, not an implementation detail.

## Planning Preconditions

- Phase 9 depends on Phase 8 in roadmap order. If Phase 8 still changes runtime API or component prop shape, finalize that first.
- `CLAUDE.md` is not present.
- `.claude/skills/` and `.agents/skills/` are not present.
- Existing multi-instance foundations are already in repo:
  - `pivot_engine/pivot_engine/runtime/dash_callbacks.py`
  - `pivot_engine/pivot_engine/runtime/session_gate.py`
  - `tests/test_dash_runtime_callbacks.py`
  - `tests/test_session_request_gate.py`
  - `tests/test_runtime_service.py`

<phase_requirements>
## Phase Requirements

| ID | Requirement | Planning Implication |
|----|-------------|----------------------|
| PKG-01 | `pip install dash-tanstack-pivot` works with zero extra config | Build one canonical wheel/sdist with correct runtime dependencies and import smoke tests |
| PKG-02 | `pip install dash-tanstack-pivot[redis]` works | Add and test `redis` extra dependency path end-to-end |
| PKG-03 | Publish to PyPI with semantic versioning | Tag/version policy and publish workflow with version/tag consistency checks |
| PKG-04 | `import dash_tanstack_pivot` has no missing deps | Install-time dependency graph must include Dash + backend runtime deps used by component path |
| PKG-05 | `npm run build` outputs one correct minified bundle | Enforce JS artifact contract in CI (bundle presence/uniqueness checks) |
| DOC-01 | README has 10-line working example | Add minimal quickstart that runs without heavy simulated data |
| DOC-02 | All Python props documented with types/defaults/descriptions | Create generated or maintained prop table synced to component metadata |
| DOC-03 | At least 3 example Dash apps | Add basic, hierarchical, SQL-connected examples and verify each boots |
| DOC-04 | `CHANGELOG.md` initialized | Add release-oriented changelog format and first entries |
| CI-01 | Python tests on every push | Add CI workflow job for pytest matrix |
| CI-02 | JS build on every push | Add CI workflow job for Node install/build + artifact assertions |
| CI-03 | Auto-publish to PyPI on version tag push | Add release workflow with PyPI trusted publishing or token secret |

</phase_requirements>

## Current Code Reality (What Planner Must Assume)

1. **No production CI exists yet.** `.github/workflows` is absent.
2. **No top-level README/changelog for the target package.** `README.md` and `CHANGELOG.md` are missing at repository root.
3. **Packaging is split and inconsistent.**
   - `dash_tanstack_pivot/setup.py` defines package with `install_requires=[]`.
   - Root `pyproject.toml` and `pivot_engine/pyproject.toml` define `scalable-pivot-engine` metadata with placeholder author/URLs.
4. **JS publish script is currently broken.** `dash_tanstack_pivot/package.json` uses `prepublishOnly: npm run validate-init`, but `_validate_init.py` does not exist.
5. **Bundle contract has drift risk.**
   - `dash_tanstack_pivot/dash_tanstack_pivot/dash_tanstack_pivot.min.js` exists.
   - `.min.js.map` is referenced in Python package metadata and `MANIFEST.in`, but file is currently missing.
6. **Multi-instance infrastructure already exists and is test-backed.**
   - Request context includes `session_id`, `client_instance`, `state_epoch`, `window_seq`.
   - `SessionRequestGate` keys by `(session_id, client_instance)` and supports optional Redis backing.
   - Tests explicitly cover callback registration idempotence and instance isolation.
7. **Table-scoped requests are already implemented in runtime paths.**
   - Frontend sends `table` in viewport payload and drill-through query params.
   - Runtime context resolves `table` and passes it to adapter requests.

## Standard Stack

### Packaging and Release

| Tool | Use in Phase 9 | Why |
|------|----------------|-----|
| `setuptools` + `python -m build` | Build wheel/sdist | Already aligned with current project structure |
| `twine check` | Verify metadata artifacts | Catch broken long description/metadata before publish |
| PEP 621 `pyproject.toml` | Canonical package metadata | Single source for version/dependencies |
| Git tags (`vX.Y.Z`) | Semantic release trigger | Simple and auditable semver contract |

### CI/CD

| Tool | Use in Phase 9 | Why |
|------|----------------|-----|
| GitHub Actions | Push CI and tag release automation | Required by CI-01/02/03 |
| `actions/setup-python` / `actions/setup-node` | Reproducible Python/Node jobs | Standard setup for matrix builds |
| `pypa/gh-action-pypi-publish` | PyPI publication | Standard trusted publishing path |

### Documentation and Examples

| Artifact | Use in Phase 9 | Why |
|----------|----------------|-----|
| Root `README.md` | Install, quickstart, props, multi-instance section | Satisfies DOC-01/02 and adoption goals |
| `examples/` or `dash_presentation/examples/` | 3 runnable Dash apps | Satisfies DOC-03 with executable evidence |
| `CHANGELOG.md` | Release history | Satisfies DOC-04 and publish traceability |

## Architecture Patterns

### Pattern 1: Define One Canonical Distribution Boundary

**Recommendation:** publish a single distribution named `dash-tanstack-pivot` that installs both:
- `dash_tanstack_pivot` (Dash component package)
- `pivot_engine` (runtime/backend support used by examples and callbacks)

This avoids forcing users to discover and install a second package manually, and keeps PKG-01 honest.

If dual-package publishing is chosen instead, it must still make `pip install dash-tanstack-pivot` fully functional by declaring and validating dependency installation of the backend package.

### Pattern 2: Single Source of Truth for Versioning

Use one canonical version field and enforce:
1. Git tag is `vX.Y.Z`.
2. Package version equals `X.Y.Z`.
3. Dash component package metadata and JS `package.json` version remain synchronized.

Do this with an explicit CI check step before publish. Do not rely on manual visual checks.

### Pattern 3: Multi-Instance Safety as a Public Contract

Document and test these invariants explicitly:
1. **Stable per-instance identity:** each component instance has unique `id`, `session_id`, and `client_instance`.
2. **Table-scoped backend requests:** every request path (viewport, drill-through) uses the instance's target table.
3. **Filter isolation:** filters/sorting/expanded state from instance A cannot mutate instance B.
4. **Concurrency correctness:** stale response rejection works independently per instance.

This contract already exists in code; Phase 9 must expose it in docs/examples/tests.

### Pattern 4: Example Architecture Must Prove Isolation

Add three documented apps:
1. **Basic DataFrame app** (minimal quickstart).
2. **Hierarchical app** (row expansion, totals).
3. **SQL-connected app** (connection string/table).

At least one example must run **two pivot instances in one layout** with distinct IDs, table names, callback wiring, and drill stores to prove multi-instance safety.

### Pattern 5: CI as a Release Gate, Not Just a Lint Job

Use separate jobs with explicit gates:
1. Python test job (`CI-01`).
2. JS build + bundle assertion job (`CI-02`).
3. Package build/import smoke job for `PKG-01/PKG-04`.
4. Tag-publish job (`CI-03`) dependent on all above.

### Pattern 6: Redis Extra Must Be Runtime-Tested

`redis` is optional in current runtime code paths (`SessionRequestGate`, `RedisCache`).
Phase 9 should add:
- install smoke for `[redis]`
- minimal runtime smoke that imports redis-backed paths without crashing

This turns PKG-02 into tested behavior, not metadata only.

## Multi-Instance Safety Requirements for Phase 9 Output

Phase 9 planning must include artifacts that explicitly prove:

1. Multiple pivot components on one page/app are supported and documented.
2. Per-instance identity (`id`, `session_id`, `client_instance`) is stable and unique.
3. Backend requests are table-scoped per instance (`table` always carried through request context).
4. Filter and sort state stay isolated by instance.
5. Concurrency and performance expectations are documented for 2+ simultaneous instances.
6. Packaging/docs include at least one multi-instance runnable example and one automated isolation test command.

## Don't Hand-Roll

| Problem | Do Not Build | Use Instead | Why |
|---------|---------------|-------------|-----|
| PyPI upload protocol | Custom curl upload scripts | `pypa/gh-action-pypi-publish` | Safer, standard, audited |
| Release version inference | Ad-hoc shell parsing | Explicit tag/version check step in CI | Deterministic semver enforcement |
| Props docs by memory | Manual unsynced prose | Generate table from component metadata/signature | Prevent drift |
| Multi-instance correctness claims | Screenshot-only proof | Automated tests (`test_dash_runtime_callbacks`, `test_session_request_gate`, integration smoke) | Prevent regressions |
| Bundle correctness checks | Assume webpack success means valid package | Assert expected output files exist and are unique | Catches artifact drift |

## Common Pitfalls

### Pitfall 1: Metadata Fragmentation

**What goes wrong:** version/dependency metadata diverges between root `pyproject`, `pivot_engine/pyproject`, `setup.py`, and `package.json`.

**Why it happens:** no single canonical packaging surface.

**How to avoid:** designate one canonical package manifest and make CI fail on mismatch.

### Pitfall 2: Broken Publish Hook

**What goes wrong:** `npm prepublishOnly` fails because `_validate_init.py` is missing.

**How to avoid:** either add the script or remove/replace the hook before wiring CI release steps.

### Pitfall 3: Import Smoke Fails in Clean Environments

**What goes wrong:** `import dash_tanstack_pivot` fails after wheel install because runtime deps were not declared.

**How to avoid:** add wheel install + import smoke in CI from a clean env.

### Pitfall 4: Multi-Instance Cross-Talk via Non-Unique IDs

**What goes wrong:** persistence keys and callback channels collide when component IDs are reused or omitted.

**How to avoid:** docs must require unique `id` per instance and show callback wiring with distinct stores/modal IDs.

### Pitfall 5: Table Scope Leakage in Examples

**What goes wrong:** docs show one global adapter/table and imply unsafe sharing patterns.

**How to avoid:** example code should pass explicit `table` per component and demonstrate separate instance configs.

### Pitfall 6: Bundle Artifact Drift

**What goes wrong:** build passes but produced files do not match package metadata (`.map` referenced but missing, extra chunks, stale assets).

**How to avoid:** CI should verify expected artifact set and fail on missing/extra critical assets.

### Pitfall 7: Concurrency Claims Without Load Evidence

**What goes wrong:** docs claim multi-instance support without testing rapid concurrent viewport requests.

**How to avoid:** add focused concurrency smoke test using two instances with interleaved requests.

## Code Examples

### Example 1: Packaging Extras (PEP 621)

```toml
[project]
name = "dash-tanstack-pivot"
version = "0.1.0"
dependencies = [
  "dash>=2.9.0",
  "duckdb>=0.8.0",
  "pyarrow>=10.0.0",
  "ibis-framework>=4.0.0",
  "pandas>=1.5.0"
]

[project.optional-dependencies]
redis = ["redis>=5.0.0"]
```

### Example 2: Multi-Instance Dash Wiring (Docs Must Include)

```python
from dash import Dash, dcc, html, dash_table
from dash_tanstack_pivot import DashTanstackPivot
from pivot_engine.runtime import DashPivotInstanceConfig, register_dash_callbacks_for_instances

app = Dash(__name__)
app.layout = html.Div([
    DashTanstackPivot(id="pivot-a", table="sales_a", serverSide=True),
    DashTanstackPivot(id="pivot-b", table="sales_b", serverSide=True),
    dcc.Store(id="drill-a"),
    dcc.Store(id="drill-b"),
    html.Div(id="modal-a"),
    html.Div(id="modal-b"),
    dash_table.DataTable(id="table-a"),
    dash_table.DataTable(id="table-b"),
    html.Button("close", id="close-a"),
    html.Button("close", id="close-b"),
])

instances = [
    DashPivotInstanceConfig(
        pivot_id="pivot-a", drill_store_id="drill-a",
        drill_modal_id="modal-a", drill_table_id="table-a", close_drill_id="close-a"
    ),
    DashPivotInstanceConfig(
        pivot_id="pivot-b", drill_store_id="drill-b",
        drill_modal_id="modal-b", drill_table_id="table-b", close_drill_id="close-b"
    ),
]

register_dash_callbacks_for_instances(app, get_runtime_service, instances, debug=False)
```

### Example 3: CI Workflow Skeleton

```yaml
name: ci
on:
  push:
    branches: ["**"]
  pull_request:

jobs:
  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: python -m pip install -U pip
      - run: python -m pip install -e .[redis]
      - run: pytest tests -q

  js-build:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: dash_tanstack_pivot
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: "20" }
      - run: npm ci
      - run: npm run build
      - run: test -f dash_tanstack_pivot/dash_tanstack_pivot.min.js
```

### Example 4: Release Publish on Tags

```yaml
name: publish
on:
  push:
    tags:
      - "v*.*.*"

jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: python -m pip install -U build twine
      - run: python -m build
      - run: twine check dist/*
      - uses: pypa/gh-action-pypi-publish@release/v1
```

## Recommended Plan Slices (for planner conversion)

1. **09-01 Packaging foundation**
   - Canonical package metadata and dependency graph.
   - Fix broken JS publish hook.
   - Build + wheel install + import smoke for PKG-01/02/04/05.
2. **09-02 Docs and examples**
   - Root README with 10-line quickstart and full prop table.
   - 3 runnable example apps including one multi-instance app.
   - Initialize `CHANGELOG.md`.
3. **09-03 CI/CD and release automation**
   - Push CI for Python tests and JS build.
   - Package artifact checks and multi-instance smoke tests.
   - Tag-triggered PyPI publish with semver gate.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Python tests | `pytest` |
| JS build verification | `npm run build` in `dash_tanstack_pivot/` |
| Package verification | `python -m build`, wheel install, import smoke |
| CI platform | GitHub Actions |

### Requirement to Test Map

| Req ID | Evidence Type | Command / Check |
|--------|----------------|-----------------|
| PKG-01 | install smoke | `pip install dist/*.whl` then `python -c "import dash_tanstack_pivot"` |
| PKG-02 | extras smoke | `pip install 'dist/*.whl[redis]'` or source install `.[redis]` |
| PKG-03 | semver publish gate | tag/version equality check + publish workflow on `v*.*.*` |
| PKG-04 | clean import check | `python -c "import dash_tanstack_pivot"` in fresh env |
| PKG-05 | bundle artifact check | `npm run build` and assert single expected minified bundle file |
| DOC-01 | doc review + run | quickstart snippet executed in CI smoke job |
| DOC-02 | docs completeness check | prop table generated from component metadata/signature |
| DOC-03 | example runtime checks | run three example apps in smoke mode (import + layout + callback registration) |
| DOC-04 | file presence/content | `CHANGELOG.md` exists with initial release entries |
| CI-01 | workflow pass | push triggers Python test job |
| CI-02 | workflow pass | push triggers JS build job |
| CI-03 | workflow pass | tag triggers publish job after gates pass |

### Multi-Instance Safety Checks (Mandatory)

- `pytest tests/test_dash_runtime_callbacks.py -q`
- `pytest tests/test_session_request_gate.py -q`
- `pytest tests/test_runtime_service.py -q`
- Add one example-level smoke test asserting:
  - two instances, different table names,
  - independent filters,
  - no cross-instance callback outputs.

## Sources

### Primary (HIGH confidence)

- `.planning/REQUIREMENTS.md`
- `.planning/ROADMAP.md`
- `.planning/STATE.md`
- `dash_tanstack_pivot/setup.py`
- `dash_tanstack_pivot/package.json`
- `dash_tanstack_pivot/webpack.config.js`
- `dash_tanstack_pivot/dash_tanstack_pivot/__init__.py`
- `dash_tanstack_pivot/src/lib/components/DashTanstackPivot.react.js`
- `pivot_engine/pivot_engine/runtime/dash_callbacks.py`
- `pivot_engine/pivot_engine/runtime/session_gate.py`
- `pivot_engine/pivot_engine/runtime/service.py`
- `tests/test_dash_runtime_callbacks.py`
- `tests/test_session_request_gate.py`
- `tests/test_runtime_service.py`
- `.planning/codebase/INTEGRATIONS.md`

### Secondary (MEDIUM confidence)

- PyPA standard workflow conventions (`python -m build`, `twine check`, GitHub publish action).

## Metadata

**Confidence breakdown:**
- Packaging blockers: HIGH (direct file evidence)
- Multi-instance behavior: HIGH (code + tests)
- CI/CD strategy: HIGH (standard patterns, no repo-specific blockers)
- Release semantics details: MEDIUM-HIGH (depends on final versioning policy choice)

**Research date:** 2026-03-15
**Valid until:** 2026-04-15 (or until packaging boundary decisions change)