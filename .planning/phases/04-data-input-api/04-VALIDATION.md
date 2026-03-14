---
phase: 4
slug: data-input-api
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >= 7.0.0 (already installed) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` (root) |
| **Quick run command** | `pytest tests/test_data_input.py -x -q` |
| **Full suite command** | `pytest tests/ pivot_engine/tests/ -x -q` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_data_input.py -x -q`
- **After every plan wave:** Run `pytest tests/ pivot_engine/tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | API-01, API-02, API-03, API-04, API-05, API-06 | unit | `pytest tests/test_data_input.py -x -q` | ❌ W0 | ⬜ pending |
| 04-02-01 | 02 | 2 | API-01, API-02, API-03, API-04, API-05, API-06 | unit | `pytest tests/test_data_input.py -x -q` | ❌ W0 | ⬜ pending |
| 04-03-01 | 03 | 3 | API-01, API-02, API-03, API-04, API-05 | unit | `pytest tests/test_data_input.py -x -q` | ❌ W0 | ⬜ pending |
| 04-03-02 | 03 | 3 | API-01–API-06 | integration | `pytest tests/ pivot_engine/tests/ -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_data_input.py` — covers API-01 through API-06; needs polars as optional dep (skip with `pytest.importorskip("polars")`)
- [ ] `pivot_engine/pivot_engine/data_input.py` — `DataInputNormalizer` class + `DataInputError`

*Existing conftest.py at repo root handles sys.path — no new conftest needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
