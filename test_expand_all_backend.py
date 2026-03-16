import pytest
import asyncio
from pivot_engine.hierarchical_scroll_manager import HierarchicalVirtualScrollManager
from pivot_engine.types.pivot_spec import PivotSpec, Measure
from pivot_engine.planner.ibis_planner import IbisPlanner
from pivot_engine.cache.memory_cache import MemoryCache
from pivot_engine.materialized_hierarchy_manager import MaterializedHierarchyManager
import ibis
import pandas as pd

@pytest.fixture
def con():
    con = ibis.duckdb.connect(":memory:")
    df = pd.DataFrame({
        "region": ["NA", "NA", "EU", "EU", "AS", "AS"],
        "country": ["US", "CA", "FR", "DE", "CN", "JP"],
        "sales": [100, 200, 150, 250, 300, 400]
    })
    con.create_table("sales", df)
    return con

@pytest.fixture
def ibis_planner(con):
    return IbisPlanner(con)

@pytest.fixture
def mock_cache():
    return MemoryCache()

@pytest.fixture
def materialized_hierarchy_manager(con, mock_cache):
    return MaterializedHierarchyManager(con, mock_cache)


def _visible_paths(rows):
    return [row["_path"] for row in rows]

@pytest.mark.asyncio
async def test_expand_all_wildcard(ibis_planner, mock_cache, materialized_hierarchy_manager):
    scroll_manager = HierarchicalVirtualScrollManager(ibis_planner, mock_cache, materialized_hierarchy_manager)
    
    spec = PivotSpec(
        table="sales",
        rows=["region", "country"],
        measures=[Measure(field="sales", agg="sum", alias="total_sales")],
        filters=[]
    )
    scroll_manager.spec = spec
    
    materialized_hierarchy_manager.create_materialized_hierarchy(spec)
    
    # Test with wildcard expansion [['__ALL__']]
    results = scroll_manager.get_visible_rows_hierarchical(
        spec=spec,
        start_row=0,
        end_row=100,
        expanded_paths=[["__ALL__"]]
    )
    
    # With 3 regions and 2 countries each, we expect 3 (level 0) + 6 (level 1) = 9 rows total
    # But wait, it's hierarchical. 
    # NA (expanded) -> US, CA
    # EU (expanded) -> FR, DE
    # AS (expanded) -> CN, JP
    # Total = 3 + 6 = 9
    
    assert len(results) == 9
    
    # Check that all rows have is_expanded=True (except leaves)
    # Actually my fix sets is_expanded=True for EVERY row returned if expand_all is True.
    for row in results:
        # If it's not a leaf, it should be expanded.
        # In this simple case, level 0 (region) rows are NOT leaves.
        # Level 1 (country) rows ARE leaves.
        if row.get("__level") == 0:
            assert row.get("is_expanded") is True


def test_expand_all_then_selective_expansion_keeps_row_totals_consistent(
    ibis_planner, mock_cache, materialized_hierarchy_manager
):
    scroll_manager = HierarchicalVirtualScrollManager(
        ibis_planner, mock_cache, materialized_hierarchy_manager
    )

    spec = PivotSpec(
        table="sales",
        rows=["region", "country"],
        measures=[Measure(field="sales", agg="sum", alias="total_sales")],
        filters=[],
    )

    materialized_hierarchy_manager.create_materialized_hierarchy(spec)

    expand_all_rows = scroll_manager.get_visible_rows_hierarchical(
        spec=spec,
        start_row=0,
        end_row=100,
        expanded_paths=[["__ALL__"]],
    )
    selective_rows = scroll_manager.get_visible_rows_hierarchical(
        spec=spec,
        start_row=0,
        end_row=100,
        expanded_paths=[["NA"]],
    )

    assert len(expand_all_rows) == 9
    assert scroll_manager.get_total_visible_row_count(spec, [["__ALL__"]]) == 9

    expected_selective_paths = ["AS", "EU", "NA", "NA|||CA", "NA|||US"]
    assert _visible_paths(selective_rows) == expected_selective_paths
    assert len(selective_rows) == len(expected_selective_paths)
    assert scroll_manager.get_total_visible_row_count(spec, [["NA"]]) == len(expected_selective_paths)


def test_collapse_after_expand_all_restores_unique_top_level_siblings(
    ibis_planner, mock_cache, materialized_hierarchy_manager
):
    scroll_manager = HierarchicalVirtualScrollManager(
        ibis_planner, mock_cache, materialized_hierarchy_manager
    )

    spec = PivotSpec(
        table="sales",
        rows=["region", "country"],
        measures=[Measure(field="sales", agg="sum", alias="total_sales")],
        filters=[],
    )

    materialized_hierarchy_manager.create_materialized_hierarchy(spec)

    scroll_manager.get_visible_rows_hierarchical(
        spec=spec,
        start_row=0,
        end_row=100,
        expanded_paths=[["__ALL__"]],
    )
    collapsed_rows = scroll_manager.get_visible_rows_hierarchical(
        spec=spec,
        start_row=0,
        end_row=100,
        expanded_paths=[],
    )

    collapsed_paths = _visible_paths(collapsed_rows)

    assert collapsed_paths == ["AS", "EU", "NA"]
    assert len(collapsed_paths) == len(set(collapsed_paths))
    assert len(collapsed_rows) == scroll_manager.get_total_visible_row_count(spec, [])
