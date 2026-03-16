---
phase: 02
slug: data-correctness-bugs
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-13
---

# Phase 02 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml`, `pivot_engine/pyproject.toml` |
| **Quick run command** | `python -m pytest tests/test_visual_totals.py tests/test_frontend_filters.py tests/test_frontend_contract.py pivot_engine/tests/test_controller.py test_expand_all_backend.py -q --tb=line` |
| **Full suite command** | `python -m pytest tests/ test_expand_all_backend.py test_filtering.py pivot_engine/tests/ pivot_engine/test_arrow_conversion.py pivot_engine/test_async_changes.py pivot_engine/test_cursor_simple.py pivot_engine/test_scalable_async_changes.py pivot_engine/test_totals_demo.py -v --tb=short` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_visual_totals.py tests/test_frontend_filters.py tests/test_frontend_contract.py pivot_engine/tests/test_controller.py test_expand_all_backend.py -q --tb=line`
- **After every plan wave:** Run `python -m pytest tests/ test_expand_all_backend.py test_filtering.py pivot_engine/tests/ pivot_engine/test_arrow_conversion.py pivot_engine/test_async_changes.py pivot_engine/test_cursor_simple.py pivot_engine/test_scalable_async_changes.py pivot_engine/test_totals_demo.py -v --tb=short`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 20 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | BUG-01..BUG-06 | regression | `python -m pytest tests/test_visual_totals.py tests/test_frontend_contract.py tests/test_frontend_filters.py pivot_engine/tests/test_controller.py test_expand_all_backend.py -q --tb=line` | yes | green |
| 02-02-01 | 02 | 2 | BUG-01 | unit/integration | `python -m pytest tests/test_visual_totals.py pivot_engine/tests/test_controller.py -q --tb=line` | yes | green |
| 02-02-02 | 02 | 2 | BUG-06 | unit/integration | `python -m pytest tests/test_frontend_contract.py pivot_engine/tests/test_controller.py -q --tb=line` | yes | green |
| 02-03-01 | 03 | 2 | BUG-03 | integration | `python -m pytest tests/test_frontend_contract.py tests/test_frontend_filters.py -q --tb=line` | yes | green |
| 02-03-02 | 03 | 2 | BUG-04 | integration | `python -m pytest tests/test_frontend_contract.py tests/test_frontend_filters.py -q --tb=line` | yes | green |
| 02-04-01 | 04 | 3 | BUG-02 | integration | `python -m pytest tests/test_frontend_contract.py test_expand_all_backend.py -q --tb=line` | yes | green |
| 02-04-02 | 04 | 3 | BUG-05 | integration | `python -m pytest tests/test_frontend_contract.py tests/test_frontend_filters.py test_expand_all_backend.py -q --tb=line` | yes | green |

*Status: pending, green, red, flaky*

---

## Wave 0 Requirements

- [x] `tests/test_visual_totals.py` - extend coverage to avg/count/min/max totals
- [x] `tests/test_frontend_contract.py` - add refresh/expansion/filter/sort persistence sequences
- [x] `pivot_engine/tests/test_controller.py` - add pivoted-column sorting and totals regressions

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Grand total row stays visually stable during repeated UI interactions | BUG-02 | Automated tests verify payload stability; manual smoke test only checks browser-visible flicker | Run the Dash demo, apply filter, sort, expand, then scroll; confirm a single stable grand total row remains visible |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 20s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** complete
