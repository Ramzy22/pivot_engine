---
phase: 5
slug: field-zone-ui
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x + pytest-asyncio |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` — testpaths = `["tests"]` |
| **Quick run command** | `pytest tests/test_field_zone_ui.py -x` |
| **Full suite command** | `pytest tests/ -x --continue-on-collection-errors` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_field_zone_ui.py -x`
- **After every plan wave:** Run `pytest tests/ -x --continue-on-collection-errors`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | FIELD-01, FIELD-02 | grep/manual | `grep -c "filter" dash_tanstack_pivot/src/lib/components/DashTanstackPivot.react.js` | ✅ | ⬜ pending |
| 05-01-02 | 01 | 1 | FIELD-03 | grep | `grep -c "min\|max" dash_tanstack_pivot/src/lib/components/DashTanstackPivot.react.js` | ✅ | ⬜ pending |
| 05-02-01 | 02 | 2 | FIELD-04, FIELD-05, FIELD-06 | unit | `pytest tests/test_field_zone_ui.py -x` | ❌ W0 | ⬜ pending |
| 05-02-02 | 02 | 2 | FIELD-02–FIELD-06 | integration | `pytest tests/ -x --continue-on-collection-errors` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_field_zone_ui.py` — covers FIELD-02, FIELD-03, FIELD-05, FIELD-06 via Python adapter contract tests:
  - `test_filter_zone_drop_updates_request` — filter field in filters dict produces correct filtered pivot response
  - `test_min_max_agg_supported_by_planner` — `valConfigs=[{field:'sales', agg:'min'}]` and `agg:'max'` produce non-empty results
  - `test_initial_rowfields_used_in_request` — adapter produces correct pivot when initialized with pre-set field configs

*FIELD-01 (four zones render) and FIELD-04 (remove field updates pivot) are verified via grep on JS source and manual browser smoke test respectively — no new test file needed for those.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Four labeled drop zones visible in sidebar | FIELD-01 | Pure React rendering — no Python assertion possible | Run `dash_presentation/app.py`, open browser, confirm Rows/Columns/Values/Filters zones appear |
| Removing a field updates pivot without reload | FIELD-04 | React state update — no server round-trip to assert | Run app, drag a field in, click remove chip, confirm pivot re-renders |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
