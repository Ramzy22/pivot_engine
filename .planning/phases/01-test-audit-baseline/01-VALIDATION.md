# Phase 01 Validation Architecture

**Phase:** 01-test-audit-baseline
**Generated from:** 01-RESEARCH.md § Validation Architecture
**Date:** 2026-03-13

---

## Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 |
| Config file | `pivot_engine/pyproject.toml` (for `pivot_engine/tests/`); root `pyproject.toml` for root tests |
| Quick run command | `cd pivot_engine && pytest tests/ -q --tb=line` |
| Full suite command | `cd pivot_engine && pytest tests/ test_arrow_conversion.py test_async_changes.py test_cursor_simple.py test_scalable_async_changes.py test_totals_demo.py -v --tb=short && cd .. && pytest tests/ test_expand_all_backend.py test_filtering.py -v --tb=short` |

---

## Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| QUAL-01 | All tests collected and passing | integration (audit) | `pytest pivot_engine/tests/ tests/ -v --tb=short` | yes (existing tests) |
| QUAL-02 | Coverage report generated and documented | reporting | `cd pivot_engine && pytest tests/ --cov=pivot_engine --cov-report=term-missing` | no — Wave 0 requires `pip install pytest-cov` first |

---

## Sampling Rate

- **Per task commit:** `cd pivot_engine && pytest tests/ -q --tb=line`
- **Per wave merge:** Full suite command from above
- **Phase gate:** Full suite green + `TEST_BASELINE.md` written before `/gsd:verify-work`

---

## Wave 0 Gaps

These must be resolved before the requirement test commands above can be executed:

- [ ] `pip install pytest-cov httpx` — required for QUAL-02 and to unblock `test_features_impl.py` collection
- [ ] `conftest.py` at repo root — required to run `pivot_engine/tests/` from repo root and to collect `test_filtering.py`, `test_expand_all_backend.py`
- [ ] Resolve or skip 4 collection errors (`test_advanced_planning.py`, `test_config_main.py`, `test_diff_engine_enhancements.py`, `test_microservices.py`) — required before QUAL-01 can be considered complete
- [ ] `.planning/phases/01-test-audit-baseline/TEST_BASELINE.md` — the Phase 1 deliverable file does not exist yet
