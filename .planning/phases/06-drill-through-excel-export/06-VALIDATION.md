---
phase: 6
slug: drill-through-excel-export
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-15
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest ≥7.0.0 + pytest-asyncio ≥0.20.0 |
| **Config file** | `pivot_engine/pyproject.toml` |
| **Quick run command** | `python -m pytest tests/test_drill_through.py -x` |
| **Full suite command** | `python -m pytest tests/ -x --ignore=tests/test_frontend_contract.py` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_drill_through.py -x`
- **After every plan wave:** Run `python -m pytest tests/ -x --ignore=tests/test_frontend_contract.py`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| Task 1 | 06-01 | 1 | EXPORT-05 | unit | `python -c "txt=open('.planning/REQUIREMENTS.md').read(); assert 'EXPORT-05' in txt"` | ❌ Wave 0 | pending |
| Task 2 | 06-01 | 1 | DRILL-03..06, EXPORT-02..05 | unit | `python -m pytest tests/test_drill_through.py tests/test_export.py --tb=no -q` | ❌ Wave 0 | pending |
| Task 1 | 06-02 | 2 | EXPORT-01..05 | integration | `python -m pytest tests/test_export.py -x --tb=short -q` | ❌ Wave 0 | pending |
| Task 2 | 06-02 | 2 | DRILL-03..06 | integration | `python -m pytest tests/test_drill_through.py -x --tb=short -q` | ❌ Wave 0 | pending |
| Task 1 | 06-03 | 2 | EXPORT-01..05 | build | `npm run build 2>&1 \| tail -20` | ✓ exists | pending |
| Task 1 | 06-04 | 3 | DRILL-01..02 | file | `test -f "dash_tanstack_pivot/src/lib/components/DrillThroughModal.js" && echo "exists"` | ❌ Wave 0 | pending |
| Task 2 | 06-04 | 3 | DRILL-01..02 | build | `npm run build 2>&1 \| tail -10` | ✓ exists | pending |

---

## Wave 0 Gaps (created by 06-01)

- [ ] `tests/test_drill_through.py` — covers DRILL-03, DRILL-04, DRILL-05, DRILL-06
- [ ] `tests/test_export.py` — covers EXPORT-02, EXPORT-03, EXPORT-04, EXPORT-05
- [ ] `dash_tanstack_pivot/src/lib/components/DrillThroughModal.js` — created in 06-04

---

## Manual-Only Requirements

| Req | Reason |
|-----|--------|
| DRILL-01 | Cell click trigger — requires browser |
| DRILL-02 | Modal render — requires browser |
| EXPORT-01 | File download API — requires browser |

These are verified by `/gsd:verify-work` UAT, not automated tests.

---

## Requirements → Test Coverage

| Req ID | Behavior | Automated | Plan |
|--------|----------|-----------|------|
| DRILL-01 | Cell click opens modal | manual-only | 06-04 |
| DRILL-02 | Modal shows source rows | manual-only | 06-04 |
| DRILL-03 | `/api/drill-through` returns rows | `test_drill_through.py::test_endpoint_returns_rows` | 06-02 |
| DRILL-04 | Pagination slices correctly | `test_drill_through.py::test_pagination` | 06-02 |
| DRILL-05 | Sort + text filter applied | `test_drill_through.py::test_sort_and_filter` | 06-02 |
| DRILL-06 | Pivot coordinate filters applied | `test_drill_through.py::test_coordinate_filters` | 06-02 |
| EXPORT-01 | Export button triggers download | manual-only | 06-03 |
| EXPORT-02 | Multi-level headers in xlsx | `test_export.py::test_xlsx_multi_level_headers` | 06-03 |
| EXPORT-03 | Row hierarchy indentation | `test_export.py::test_xlsx_row_indentation` | 06-03 |
| EXPORT-04 | Grand totals included | `test_export.py::test_xlsx_includes_totals` | 06-03 |
| EXPORT-05 | >500k rows → csv branch | `test_export.py::test_export_format_threshold` | 06-03 |
