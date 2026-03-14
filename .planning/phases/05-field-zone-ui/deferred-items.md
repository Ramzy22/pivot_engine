# Deferred Items

## 2026-03-14

- Full-suite verification for `05-02-PLAN.md` exposed pre-existing failures outside this plan's write scope:
  - `tests/test_frontend_contract.py::test_sorting`
  - `tests/test_frontend_filters.py::test_floating_filter_backend_logic`
  - `tests/test_frontend_filters.py::test_filter_and_sort_state_survives_repeated_requests`
- Symptoms:
  - grouped sort ordering is not descending as asserted
  - filtered responses include `nan` rows that break string assertions in frontend filter tests
- These were not modified by this plan, so they were logged here rather than auto-fixed inline.
