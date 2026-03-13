# Test Baseline — Phase 1
**Date:** 2026-03-13
**Established before any production code changes**

## Summary

| Category | Count |
|----------|-------|
| pytest tests: passed | 55 |
| pytest tests: skipped | 13 |
| pytest tests: failed | 0 |
| collection errors | 0 |
| script tests: passed (exit 0) | 5 |
| script tests: failed (non-zero exit) | 10 |

**All 55 runnable pytest tests pass. Zero failures. Zero collection errors.**

## pytest Test Files

### Passing Files

| File | Tests | Notes |
|------|-------|-------|
| `tests/test_frontend_contract.py` | 4 | Frontend contract (hierarchy, expansion, filter, sort) |
| `tests/test_frontend_filters.py` | 1 | Floating filter backend logic |
| `tests/test_multi_condition_filters.py` | 2 | AND composite + single condition filters |
| `tests/test_visual_totals.py` | 1 | Visual totals with measure filter |
| `test_expand_all_backend.py` | 1 | Expand-all wildcard |
| `test_filtering.py` | 1 | Filtering integration |
| `pivot_engine/tests/clickhouse_compatibility_test.py` | 3 | ClickHouse URI parsing + backend-agnostic features |
| `pivot_engine/tests/clickhouse_verification_test.py` | 1 | Backend-agnostic verification |
| `pivot_engine/tests/test_cache.py` | 6 (memory) | Cache set/get/overwrite/clear/TTL/complex — redis variants skipped |
| `pivot_engine/tests/test_cdc.py` | 7 | CDC initialization, setup, insert/update events, stream, integration, cache keys |
| `pivot_engine/tests/test_complete_implementation.py` | 2 | TanStack adapter + controller directly (scalable_features skipped) |
| `pivot_engine/tests/test_controller.py` | 5 (excl. redis) | Ibis planner: simple, filter, totals, sort, cursor pagination |
| `pivot_engine/tests/test_hierarchical_managers.py` | 1 | Ibis-based hierarchical scroll with pagination |
| `pivot_engine/tests/test_scalable_pivot.py` | 5 | Scalable controller: basic, materialized hierarchy, virtual scroll, progressive load, batch load |
| `pivot_engine/tests/test_streaming_incremental.py` | 10 | Streaming processor creation, jobs, views, updates, edge cases |
| `pivot_engine/test_arrow_conversion.py` | 1 | Arrow to JSON conversion |
| `pivot_engine/test_async_changes.py` | 1 | Async CDC change detection |
| `pivot_engine/test_cursor_simple.py` | 1 | Cursor pagination (simple) |
| `pivot_engine/test_scalable_async_changes.py` | 1 | Scalable async CDC changes |
| `pivot_engine/test_totals_demo.py` | 1 | Totals feature |

### Skipped Files / Tests

| File | Test(s) Skipped | Skip Reason |
|------|-----------------|-------------|
| `pivot_engine/tests/test_advanced_planning.py` | entire file (module skip) | `pivot_engine.planner.sql_planner` module does not exist — aspirational test for unimplemented planner |
| `pivot_engine/tests/test_config_main.py` | entire file (module skip) | `structlog` package not installed — transitive import chain blocks collection |
| `pivot_engine/tests/test_diff_engine_enhancements.py` | entire file (module skip) | `MultiDimensionalTilePlanner` class not in `diff_engine` — aspirational test |
| `pivot_engine/tests/test_features_impl.py` | entire file (module skip) | `structlog` package not installed — transitive via `complete_rest_api` |
| `pivot_engine/tests/test_microservices.py` | entire file (module skip) | `pivot_engine.pivot_microservices` module does not exist — aspirational test |
| `pivot_engine/tests/test_cache.py` | 6 redis variants | `fakeredis` not installed in test environment |
| `pivot_engine/tests/test_controller.py` | `test_controller_with_redis_cache` | `fakeredis` not installed in test environment |
| `pivot_engine/tests/test_complete_implementation.py` | `test_scalable_features` | DuckDB connection concurrency bug: `run_materialized_hierarchy` starts background thread holding the DuckDB connection; `run_pruned_hierarchical_pivot` then fails with `InvalidInputException: closed pending query result` — deferred to Phase 2 |

## Script-Style Test Files

These files use `asyncio.run()` / direct execution and are not pytest-runnable.
They are documented here for completeness but are not part of the pytest baseline.

| File | Location | Exit Code | Notes |
|------|----------|-----------|-------|
| `test_expand_all.py` | root | 1 | `pivot_engine.scalable_pivot_controller` not on sys.path (no conftest.py for direct execution) |
| `test_fix_verification.py` | root | 1 | Same path issue |
| `test_flat_final.py` | root | 1 | `pivot_engine.controller` not on sys.path |
| `test_flat_output.py` | root | 1 | Same path issue |
| `test_ifelse.py` | root | 0 | Pass — no pivot_engine imports needed |
| `test_tanstack_communication.py` | root | 1 | `pivot_engine.scalable_pivot_controller` not on sys.path |
| `test_virtual_scroll.py` | root | 1 | Same path issue |
| `test_virtual_scroll_pivot.py` | root | 1 | Same path issue |
| `pivot_engine/test_backend.py` | pivot_engine | 1 | Missing `sales_pagination` table in DuckDB (requires test fixture setup) |
| `pivot_engine/test_connection.py` | pivot_engine | 1 | Missing `sales_pagination` table (same issue) |
| `pivot_engine/test_exact_issue.py` | pivot_engine | 1 | Ibis `IbisTypeError`: `sql` column not found in result (known API mismatch in script) |
| `pivot_engine/test_final_implementation.py` | pivot_engine | 0 | Pass |
| `pivot_engine/test_hierarchical_async.py` | pivot_engine | 0 | Pass |
| `pivot_engine/test_materialized_hierarchy.py` | pivot_engine | 0 | Pass (despite partial error message logged internally) |
| `pivot_engine/test_with_clear_cache.py` | pivot_engine | 0 | Pass |

**Summary: 5 pass, 10 fail** — root-level scripts fail because they lack the conftest.py sys.path fix that pytest applies automatically. This is expected and informational.

## JS Tests

No JS tests are configured. The `dash_tanstack_pivot/package.json` has no `test` script
and jest is not in devDependencies. JS test coverage is 0% and this is expected for Phase 1.

## Python Coverage

Measured via `pytest-cov` from `pivot_engine/` directory.
Command: `pytest tests/ --cov=pivot_engine --cov-branch --cov-report=term-missing`

| Module | Statements | Miss | Branch | BrPart | Coverage% |
|--------|------------|------|--------|--------|-----------|
| pivot_engine\\__init__.py | 4 | 0 | 0 | 0 | 100% |
| pivot_engine\\admin_api.py | 52 | 52 | 6 | 0 | 0% |
| pivot_engine\\backends\\duckdb_backend.py | 157 | 82 | 38 | 5 | 43% |
| pivot_engine\\backends\\ibis_backend.py | 179 | 131 | 68 | 9 | 25% |
| pivot_engine\\cache\\__init__.py | 0 | 0 | 0 | 0 | 100% |
| pivot_engine\\cache\\memory_cache.py | 44 | 10 | 16 | 1 | 72% |
| pivot_engine\\cache\\redis_cache.py | 52 | 37 | 8 | 0 | 25% |
| pivot_engine\\cdc\\__init__.py | 0 | 0 | 0 | 0 | 100% |
| pivot_engine\\cdc\\cdc_manager.py | 107 | 27 | 42 | 8 | 70% |
| pivot_engine\\cdc\\database_change_detector.py | 138 | 81 | 46 | 1 | 33% |
| pivot_engine\\cdc\\models.py | 8 | 0 | 0 | 0 | 100% |
| pivot_engine\\common\\__init__.py | 0 | 0 | 0 | 0 | 100% |
| pivot_engine\\common\\ibis_expression_builder.py | 280 | 190 | 194 | 29 | 27% |
| pivot_engine\\complete_rest_api.py | 356 | 356 | 14 | 0 | 0% |
| pivot_engine\\config.py | 80 | 80 | 18 | 0 | 0% |
| pivot_engine\\controller.py | 325 | 172 | 134 | 19 | 44% |
| pivot_engine\\dash_component.py | 60 | 60 | 12 | 0 | 0% |
| pivot_engine\\diff\\__init__.py | 0 | 0 | 0 | 0 | 100% |
| pivot_engine\\diff\\diff_engine.py | 392 | 240 | 170 | 17 | 32% |
| pivot_engine\\flight_server.py | 37 | 37 | 8 | 0 | 0% |
| pivot_engine\\hierarchical_scroll_manager.py | 560 | 387 | 346 | 38 | 26% |
| pivot_engine\\incremental_ui_updates.py | 140 | 140 | 62 | 0 | 0% |
| pivot_engine\\intelligent_prefetch_manager.py | 142 | 122 | 52 | 0 | 10% |
| pivot_engine\\lifecycle.py | 53 | 22 | 14 | 3 | 51% |
| pivot_engine\\main.py | 14 | 14 | 2 | 0 | 0% |
| pivot_engine\\main_complete.py | 135 | 135 | 8 | 0 | 0% |
| pivot_engine\\materialized_hierarchy_manager.py | 123 | 31 | 32 | 5 | 70% |
| pivot_engine\\observability.py | 10 | 10 | 0 | 0 | 0% |
| pivot_engine\\planner\\__init__.py | 0 | 0 | 0 | 0 | 100% |
| pivot_engine\\planner\\expression_parser.py | 34 | 27 | 16 | 0 | 14% |
| pivot_engine\\planner\\ibis_planner.py | 558 | 367 | 328 | 37 | 29% |
| pivot_engine\\progressive_loader.py | 147 | 125 | 68 | 0 | 10% |
| pivot_engine\\pruning_manager.py | 198 | 113 | 104 | 10 | 36% |
| pivot_engine\\scalable_pivot_controller.py | 451 | 259 | 158 | 17 | 38% |
| pivot_engine\\security.py | 52 | 26 | 18 | 0 | 37% |
| pivot_engine\\streaming\\__init__.py | 2 | 0 | 0 | 0 | 100% |
| pivot_engine\\streaming\\state_store.py | 27 | 6 | 0 | 0 | 78% |
| pivot_engine\\streaming\\streaming_processor.py | 351 | 190 | 204 | 22 | 39% |
| pivot_engine\\tanstack_adapter.py | 371 | 228 | 196 | 22 | 33% |
| pivot_engine\\tree.py | 347 | 222 | 132 | 18 | 32% |
| pivot_engine\\types\\pivot_spec.py | 78 | 1 | 2 | 1 | 98% |
| **TOTAL** | **6064** | **3980** | **2516** | **262** | **30%** |

**Total coverage: 30%**

### Modules with Low Coverage (< 50%)

Informational only — no gate in Phase 1.

| Module | Coverage | Reason |
|--------|----------|--------|
| `admin_api.py` | 0% | No tests exist for admin HTTP endpoints |
| `complete_rest_api.py` | 0% | Tests skipped (structlog import) |
| `config.py` | 0% | Tests skipped (structlog import chain) |
| `dash_component.py` | 0% | No unit tests for Dash component wrapper |
| `flight_server.py` | 0% | No tests for Arrow Flight server |
| `incremental_ui_updates.py` | 0% | No tests |
| `main.py` | 0% | Tests skipped (structlog) |
| `main_complete.py` | 0% | No tests |
| `observability.py` | 0% | Blocked by missing structlog; tests skipped |
| `backends/ibis_backend.py` | 25% | Ibis backend exercised only partially through controller tests |
| `common/ibis_expression_builder.py` | 27% | Large module; only filter/aggregation paths exercised |
| `hierarchical_scroll_manager.py` | 26% | Complex virtual scroll; only single-path expansion tested |
| `planner/ibis_planner.py` | 29% | Large planner; only basic pivot paths exercised |
| `planner/expression_parser.py` | 14% | Expression parser; mostly untested |
| `progressive_loader.py` | 10% | No direct tests |
| `intelligent_prefetch_manager.py` | 10% | No direct tests |

## Known Gaps & Notes

- 5 test files skip at module level due to unimplemented modules (`planner.sql_planner`, `pivot_microservices`) or missing packages (`structlog`). These were aspirational tests written before the code/packages existed.
- `test_scalable_features` is skipped: DuckDB connection reuse issue when mixing async materialize with sync pruned pivot. This is a real production bug deferred to Phase 2.
- 10 standalone scripts fail at root level because they rely on direct `python script.py` invocation without the conftest.py sys.path fix. These are debug/reproduction scripts, not formal tests.
- JS test infrastructure is absent from `dash_tanstack_pivot/` — no jest config, no test script in package.json. This is Phase 1 scope: document only.
- `asyncio: mode=Mode.STRICT` is configured in `pivot_engine/pyproject.toml`. The clickhouse_compatibility_test uses `asyncio.get_event_loop().run_until_complete()` with a DeprecationWarning; this is acceptable for Phase 1 baseline.

## Next Steps

Phase 2 (Data Correctness Bugs) may now begin. No production code was changed in Phase 1.
Key items to address in Phase 2:
- DuckDB connection concurrency fix for `test_scalable_features`
- Increase coverage on core paths (ibis_expression_builder, ibis_planner, hierarchical_scroll_manager)
