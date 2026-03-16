---
phase: 03
slug: virtual-scroll-ui-bugs
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-13
---

# Phase 03 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | `pyproject.toml`, `pivot_engine/pyproject.toml` |
| **Quick run command** | `python -m pytest tests/test_frontend_contract.py tests/test_frontend_filters.py test_expand_all_backend.py pivot_engine/tests/test_hierarchical_managers.py pivot_engine/tests/test_scalable_pivot.py -q --tb=line` |
| **Full suite command** | `python -m pytest tests/ test_expand_all_backend.py test_filtering.py pivot_engine/tests/ pivot_engine/test_arrow_conversion.py pivot_engine/test_async_changes.py pivot_engine/test_cursor_simple.py pivot_engine/test_scalable_async_changes.py pivot_engine/test_totals_demo.py -v --tb=short` |
| **Estimated runtime** | ~20 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_frontend_contract.py tests/test_frontend_filters.py test_expand_all_backend.py pivot_engine/tests/test_hierarchical_managers.py pivot_engine/tests/test_scalable_pivot.py -q --tb=line`
- **After every plan wave:** Run `python -m pytest tests/ test_expand_all_backend.py test_filtering.py pivot_engine/tests/ pivot_engine/test_arrow_conversion.py pivot_engine/test_async_changes.py pivot_engine/test_cursor_simple.py pivot_engine/test_scalable_async_changes.py pivot_engine/test_totals_demo.py -v --tb=short`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 20 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | BUG-07, BUG-08, BUG-11, BUG-12 | regression | `python -m pytest tests/test_frontend_contract.py tests/test_frontend_filters.py test_expand_all_backend.py pivot_engine/tests/test_hierarchical_managers.py -q --tb=line` | yes | green |
| 03-02-01 | 02 | 2 | BUG-11, BUG-12 | integration | `python -m pytest tests/test_frontend_contract.py test_expand_all_backend.py pivot_engine/tests/test_hierarchical_managers.py -q --tb=line` | yes | green |
| 03-02-02 | 02 | 2 | BUG-07 | integration | `python -m pytest tests/test_frontend_contract.py pivot_engine/tests/test_scalable_pivot.py -q --tb=line` | yes | green |
| 03-03-01 | 03 | 2 | BUG-07, BUG-08 | integration | `python -m pytest tests/test_frontend_contract.py tests/test_frontend_filters.py -q --tb=line` | yes | green |
| 03-04-01 | 04 | 3 | BUG-09, BUG-10 | manual-plus-contract | `python -m pytest tests/test_frontend_contract.py -q --tb=line` | yes | green |
| 03-04-02 | 04 | 3 | BUG-13 | manual-plus-contract | `python -m pytest tests/test_frontend_contract.py -q --tb=line` | yes | green |

*Status: pending, green, red, flaky*

---

## Wave 0 Requirements

- [x] `tests/test_frontend_contract.py` - add virtual-scroll and expand/collapse regression sequences for stale rows and row-count continuity
- [x] `test_expand_all_backend.py` - extend hierarchy manager assertions around sibling stability and visible-row accounting
- [x] `pivot_engine/tests/test_hierarchical_managers.py` - add pagination/expand correctness coverage beyond the existing happy path

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Multi-level header groups visually align with child columns during pinning and horizontal scroll | BUG-09, BUG-10 | Existing automation covers payload contracts, not final browser geometry | Open the Dash demo with grouped columns, pin left/right columns, scroll horizontally, and confirm each header group spans exactly its visible children |
| Expand/collapse does not visibly shift or duplicate siblings in the rendered grid | BUG-11, BUG-12 | Render timing and virtualization artifacts are browser-visible concerns | Expand and collapse multiple sibling groups while scrolled mid-table; confirm no duplicate, missing, or re-ordered sibling rows appear |
| Context menu stays fully within the viewport near all edges | BUG-13 | Final placement depends on browser viewport and rendered menu dimensions | Right-click cells near the top-left, top-right, bottom-left, and bottom-right edges; confirm the menu remains fully visible |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 20s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** complete
