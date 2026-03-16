---
phase: 7
slug: column-display-ui-states
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-15
---

# Phase 7 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest` + `dash` plugin (existing) |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `pytest tests/test_frontend_contract.py tests/test_frontend_filters.py -q` |
| **Full suite command** | `pytest` |
| **Estimated runtime** | ~30-90 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_frontend_contract.py tests/test_frontend_filters.py -q`
- **After every plan wave:** Run `pytest`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 90 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | UI-03 | integration | `pytest tests/test_frontend_contract.py -k visibility` | ❌ W0 | ⬜ pending |
| 07-01-02 | 01 | 1 | UI-04 | e2e | `playwright test tests/e2e/column-resize.spec.ts` | ❌ W0 | ⬜ pending |
| 07-02-01 | 02 | 2 | UI-01 | integration | `pytest tests/test_frontend_contract.py -k pin` | ❌ W0 | ⬜ pending |
| 07-02-02 | 02 | 2 | UI-02 | integration | `pytest tests/test_frontend_contract.py -k sort` | ❌ W0 | ⬜ pending |
| 07-03-01 | 03 | 3 | UI-05 | e2e | `playwright test tests/e2e/combined-column-states.spec.ts` | ❌ W0 | ⬜ pending |
| 07-03-02 | 03 | 3 | UI-06 | visual/regression | `playwright test tests/e2e/default-density.spec.ts` | ❌ W0 | ⬜ pending |
| 07-03-03 | 03 | 3 | UI-05, UI-06 | integration/manual gate | `python -c "from pathlib import Path; t=Path('.planning/phases/07-column-display-ui-states/07-UI-STATE-CHECKLIST.md').read_text(encoding='utf-8'); req=['UI-01','UI-02','UI-03','UI-04','UI-05','UI-06']; m=[x for x in req if x not in t]; print('checklist-ready' if not m else 'missing:'+','.join(m)); raise SystemExit(0 if not m else 1)"` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/e2e/column-resize.spec.ts` - cover UI-04 (resize affordance and persistence)
- [ ] `tests/e2e/combined-column-states.spec.ts` - cover UI-05 (pinned+sorted+resized coexistence)
- [ ] `tests/e2e/default-density.spec.ts` - cover UI-06 (default width/height consistency)
- [ ] `playwright.config.ts` - browser test configuration and base URL
- [ ] `package.json` (or equivalent scripts) - add `test:e2e` command for reproducible CI/local runs

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Shadow/border visual quality at pinned boundaries | UI-01 | Visual styling quality is subjective and theme-dependent | Scroll horizontally with left/right pinning in light and dark themes; confirm separator visibility and no overlap artifacts |
| Sorted-column visual distinction quality | UI-02 | Contrast/accessibility quality requires visual review | Apply asc/desc sorts on multiple columns and verify active header emphasis remains clear |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 90s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
