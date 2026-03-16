---
phase: 9
slug: packaging-docs-ci-cd
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-15
---

# Phase 9 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | `pytest` + Dash runtime tests + npm build checks |
| **Config file** | `pyproject.toml`, `dash_tanstack_pivot/package.json` |
| **Quick run command** | `python -m pytest tests/test_session_request_gate.py -q --maxfail=1` |
| **Full suite command** | `python -m pytest -q` + `cd dash_tanstack_pivot && npm.cmd run build` |
| **Estimated runtime** | ~25 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_session_request_gate.py -q --maxfail=1`
- **After every plan wave:** Run `python -m pytest -q` and `cd dash_tanstack_pivot && npm.cmd run build`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 01 | 1 | PKG-01, PKG-04, PKG-05 | package smoke | `python -m build && python -c "import dash_tanstack_pivot"` | ✅ | ⬜ pending |
| 09-01-02 | 01 | 1 | PKG-02 | extras smoke | `pip install .[redis]` | ✅ | ⬜ pending |
| 09-02-01 | 02 | 2 | DOC-01, DOC-02 | docs contract | `python -c "from pathlib import Path; t=Path('README.md').read_text(encoding='utf-8'); req=['pip install dash-tanstack-pivot','pivot_table(','table']; m=[x for x in req if x not in t]; raise SystemExit(0 if not m else 1)"` | ✅ | ⬜ pending |
| 09-02-02 | 02 | 2 | DOC-03 | integration + multi-instance | `python -m pytest tests/test_dash_runtime_callbacks.py tests/test_session_request_gate.py tests/test_runtime_service.py -q` | ✅ | ⬜ pending |
| 09-02-03 | 02 | 2 | DOC-04 | file contract | `python -c "from pathlib import Path; p=Path('CHANGELOG.md'); raise SystemExit(0 if p.exists() and p.read_text(encoding='utf-8').strip() else 1)"` | ❌ W0 | ⬜ pending |
| 09-03-01 | 03 | 3 | CI-01, CI-02 | CI config | `python -c "from pathlib import Path; t=Path('.github/workflows/ci.yml').read_text(encoding='utf-8'); req=['pytest','npm','build']; m=[x for x in req if x not in t]; raise SystemExit(0 if not m else 1)"` | ❌ W0 | ⬜ pending |
| 09-03-02 | 03 | 3 | PKG-03, CI-03 | release gate | `python -c "from pathlib import Path; t=Path('.github/workflows/release.yml').read_text(encoding='utf-8'); req=['tags','v*.*.*','pypi']; m=[x for x in req if x not in t.lower()]; raise SystemExit(0 if not m else 1)"` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `CHANGELOG.md` - add initial release changelog scaffold (DOC-04)
- [ ] `.github/workflows/ci.yml` - push workflow with Python tests + JS build (CI-01, CI-02)
- [ ] `.github/workflows/release.yml` - semver-tag publish workflow (PKG-03, CI-03)
- [ ] Multi-instance example smoke assertion - prove two instances remain isolated by `id/session_id/client_instance/table`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real PyPI publish credentials and protected environment gate | CI-03 | Repository secrets and publisher permissions cannot be fully validated offline | Create a dry-run release tag in protected branch and confirm publish job auth path works with maintainer credentials |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
