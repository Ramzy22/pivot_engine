# Phase 2: Data Correctness Bugs - Research

**Researched:** 2026-03-13
**Domain:** Pivot correctness, TanStack request translation, hierarchical state persistence
**Confidence:** HIGH

<user_constraints>
## User Constraints

### Locked Decisions

- Phase scope is exactly BUG-01 through BUG-06 from `.planning/REQUIREMENTS.md`.
- This phase follows the Phase 1 baseline and should fix real correctness bugs before new feature work.
- Planning must preserve the existing stack: Dash frontend, TanStack adapter, Ibis planner, DuckDB-backed tests.
- No Phase 2 CONTEXT.md exists, so roadmap, requirements, state, and Phase 1 artifacts are the authoritative inputs.

### Claude's Discretion

- Exact test fixture shape and whether to use controller-level or adapter-level tests for each bug.
- Whether to fix bugs in planner, controller, adapter, virtual-scroll manager, or a combination.
- Plan granularity, so long as the plans are executable and requirement-complete.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BUG-01 | Grand total rows display correct aggregated values for all measure types | Totals are currently computed after query execution; planner visual-totals path explicitly does not handle AVG correctly |
| BUG-02 | Grand total rows do not disappear or flicker on scroll/filter changes | Totals are reconstituted in adapter/controller paths with duplicate-grand-total logic and post-query mutation |
| BUG-03 | Pivot column discovery returns complete, non-sparse column set after data changes | Dynamic columns are discovered through a separate top-N path and cached independently |
| BUG-04 | Pivot column discovery is consistent across page refreshes and filter changes | Column discovery cursor/cache behavior is tied to request translation and cache keys, not explicit refresh invalidation |
| BUG-05 | Filter state persists across row expansion, sort changes, and viewport scroll | Hierarchical and virtual-scroll requests rebuild specs from request state every call and then reconstruct rows manually |
| BUG-06 | Sort state applies server-side and does not reset on data refresh | Sort translation exists in adapter, but pivot-query sort application currently drops non-row sorts in the pivoted-columns path |

</phase_requirements>

---

## Summary

Phase 2 is concentrated in three places:

1. `pivot_engine/pivot_engine/planner/ibis_planner.py`
   This is where pre-aggregation filters, post-aggregation filters, visual totals, column discovery, and most server-side sorting decisions are made.

2. `pivot_engine/pivot_engine/scalable_pivot_controller.py`
   This is where planned queries are executed, cached, totals are appended post-query, and hierarchical/virtual-scroll entry points branch into specialized paths.

3. `pivot_engine/pivot_engine/tanstack_adapter.py`
   This is where TanStack filter/sort/grouping state is translated into `PivotSpec`, and where hierarchical rows, grand total rows, dynamic columns, and pagination metadata are rebuilt for the UI.

The codebase already has small passing tests for simple totals, visual totals, frontend filter translation, frontend sorting, and hierarchy expansion, but they do not cover the Phase 2 contract deeply enough. The biggest uncovered risks are:

- AVG grand totals are explicitly not handled in the visual-totals roll-up path.
- `scalable_pivot_controller._execute_standard_pivot_async()` computes `final_table` but returns `main_result`, which likely bypasses finalization and can destabilize totals or merged result shape.
- `tanstack_adapter.convert_tanstack_request_to_pivot_spec()` converts request state each time, but there is no dedicated persistence/invalidation contract around expansion + filter + sort + virtual scroll.
- `ibis_planner.build_pivot_query_from_columns()` only applies `spec.sort` when the sort field is in `row_dims`, which means measure-based sorts can be dropped in pivoted column mode.

Primary planning recommendation: add regression tests first, then fix the backend query semantics, then fix dynamic column discovery/invalidation, then wire state persistence across hierarchical and virtual-scroll paths.

---

## Existing Coverage

### Relevant Passing Tests

| File | What it covers | Gap vs Phase 2 |
|------|----------------|----------------|
| `pivot_engine/tests/test_controller.py` | Simple filter, totals, sort, cursor pagination | Only sum totals; no adapter/hierarchy refresh behavior |
| `tests/test_visual_totals.py` | Visual totals under post-aggregation filter | Only sum; no avg/count/min/max coverage |
| `tests/test_frontend_filters.py` | TanStack filter object translation | No persistence across expansion/sort/refresh |
| `tests/test_frontend_contract.py` | Basic hierarchy load, expansion, filtering, sorting | No state-retention sequence tests; no dynamic columns assertions |
| `test_expand_all_backend.py` | Expand-all wildcard behavior | No filter/sort interaction or scroll invalidation |

### Missing Coverage Required for Phase 2

- Grand total correctness for `avg`, `count`, `min`, `max`, not just `sum`
- Grand total stability after repeated filter changes and hierarchical/virtual-scroll requests
- Column discovery completeness after data changes and filter changes
- Column discovery consistency after repeated request rebuilds / refresh-like calls
- Sort persistence in pivoted-column mode and after refresh
- Filter persistence through expand -> sort -> scroll sequences

---

## Architecture Findings

### 1. Totals Are Added After Query Execution

`scalable_pivot_controller.py` appends totals after query execution when `metadata["needs_totals"]` is set. This means totals stability depends on controller post-processing, not only on planner correctness.

Observed risk:
- `_execute_standard_pivot_async()` computes `final_table = self.diff_engine.merge_and_finalize(...)` but returns `main_result` instead of `final_table`.
- That strongly suggests the finalize path is currently ignored in the standard path.

Impact:
- BUG-01: incorrect totals can survive even if planner output is correct.
- BUG-02: total-row shape/order can change across request paths.

### 2. Visual Totals Path Does Not Properly Support AVG

In `ibis_planner._plan_standard()`, the visual totals mode rolls filtered leaf aggregates back up to the requested level. The code comment states:

- sum/count use `sum`
- min/max use `min`/`max`
- avg is "more complex (not handled here)"

Impact:
- BUG-01 is not satisfiable without changing this logic.
- Tests must explicitly prove correct grand totals for avg/count/min/max, not just sum.

### 3. Dynamic Column Discovery Is Split From Main Query Execution

Pivoted columns are discovered first, then fed into `build_pivot_query_from_columns()`. The column-discovery result and pivot result are cached separately in `_execute_topn_pivot_async()`.

Observed risks:
- Separate cache keys mean stale column sets can survive after filter/spec changes if invalidation is incomplete.
- `column_cursor` is overloaded through `global_filter`, making horizontal pagination/state hard to reason about.

Impact:
- BUG-03 and BUG-04 likely live here rather than in the raw backend.

### 4. Sort Translation Exists, but Sort Application Is Incomplete

The adapter converts TanStack sorting into `PivotSpec.sort`, but `build_pivot_query_from_columns()` only keeps sorts whose `field` is in `row_dims`.

Impact:
- Sorting by aggregated measure or discovered pivoted column can silently drop to default order.
- This directly threatens BUG-06.

### 5. Hierarchical State Is Reconstructed, Not Preserved as a First-Class Model

`handle_hierarchical_request()` and `handle_virtual_scroll_request()` both rebuild `PivotSpec` from the incoming request, then reconstruct visible rows manually. Expansion state is carried separately as `expanded_paths`.

Observed risks:
- Any call site that omits or mutates request filters/sorting can reset server-side behavior.
- Grand total deduplication logic is hard-coded around row reconstruction.
- Virtual scroll falls back to hierarchical load on error, which can mask state mismatch instead of failing loudly.

Impact:
- BUG-02 and BUG-05 require end-to-end adapter/controller tests, not only planner tests.

---

## Recommended Plan Decomposition

### Plan 02-01
- Goal: Add missing regression coverage and shared fixtures for BUG-01 through BUG-06.
- Why first: Phase 2 is correctness-heavy and spans planner, controller, and adapter. Execution needs failing tests before refactors.

### Plan 02-02
- Goal: Fix grand totals and server-side sort semantics in planner/controller paths.
- Core targets:
  - visual totals avg/count/min/max
  - standard-path finalization return bug
  - measure sort persistence server-side

### Plan 02-03
- Goal: Fix dynamic column discovery completeness and refresh/filter invalidation.
- Core targets:
  - column discovery cache boundaries
  - complete/non-sparse discovered column sets
  - refresh consistency under changing filters/data

### Plan 02-04
- Goal: Fix filter/sort persistence across hierarchy expansion and virtual scroll.
- Core targets:
  - hierarchical request sequences
  - virtual-scroll fallback/state continuity
  - grand total stability during repeated request changes

---

## Code-Level Watchpoints

### `pivot_engine/pivot_engine/planner/ibis_planner.py`

- `_plan_standard()`
  - splits pre/post filters
  - visual totals branch handles only some aggregate roll-ups safely
- `build_pivot_query_from_columns()`
  - applies post-filters after pivoting
  - currently restricts sort application to `row_dims`
- `_build_column_values_query()`
  - separate path for discovered columns; needs regression coverage around filters and cursor behavior

### `pivot_engine/pivot_engine/scalable_pivot_controller.py`

- `_execute_standard_pivot_async()`
  - likely bug: returns `main_result` instead of `final_table`
- `_execute_topn_pivot_async()`
  - caches column discovery and pivot result independently
- `run_hierarchical_pivot_batch_load()` / `run_virtual_scroll_hierarchical()`
  - important for state persistence and stable hierarchical views

### `pivot_engine/pivot_engine/tanstack_adapter.py`

- `convert_tanstack_request_to_pivot_spec()`
  - source of filter/sort state translation
- `convert_pivot_result_to_tanstack_format()`
  - source of grand total tagging and dynamic column generation
- `handle_hierarchical_request()` / `handle_virtual_scroll_request()`
  - source of persistence bugs across expansion, sort, and scroll

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `pyproject.toml` at repo root plus `pivot_engine/pyproject.toml` |
| Quick run command | `python -m pytest tests/test_visual_totals.py tests/test_frontend_filters.py tests/test_frontend_contract.py pivot_engine/tests/test_controller.py test_expand_all_backend.py -q --tb=line` |
| Full suite command | `python -m pytest tests/ test_expand_all_backend.py test_filtering.py pivot_engine/tests/ pivot_engine/test_arrow_conversion.py pivot_engine/test_async_changes.py pivot_engine/test_cursor_simple.py pivot_engine/test_scalable_async_changes.py pivot_engine/test_totals_demo.py -v --tb=short` |

### Requirement -> Test Map

| Requirement | Primary test target |
|-------------|---------------------|
| BUG-01 | new totals-focused planner/controller tests |
| BUG-02 | new adapter hierarchy/virtual-scroll stability tests |
| BUG-03 | new dynamic-column discovery tests |
| BUG-04 | repeated-request refresh consistency tests |
| BUG-05 | adapter request-sequence persistence tests |
| BUG-06 | planner/controller sort persistence tests |

### Wave 0 Gaps

- [ ] Add explicit failing tests for avg/count/min/max totals
- [ ] Add dynamic-column discovery regression tests
- [ ] Add expansion + filter + sort + scroll sequence tests

---

## Open Questions

1. Should dynamic column discovery use explicit cache invalidation keyed on the full request state, or should it simply bypass cache on relevant spec changes?
2. Is the Phase 1 skipped `test_scalable_features` concurrency issue best treated as Phase 2 collateral, or should it stay deferred unless it blocks the targeted bugs?
3. Are row-total columns (`__RowTotal__*`) part of the expected Phase 2 behavior, or should the work stay focused on grand totals and dynamic pivot columns only?

---

## Metadata

**Confidence breakdown:**
- Planner/controller hotspots: HIGH
- Dynamic column discovery path: HIGH
- Hierarchical persistence risk: HIGH
- Exact final implementation shape: MEDIUM

**Research date:** 2026-03-13
**Valid until:** 2026-04-13
