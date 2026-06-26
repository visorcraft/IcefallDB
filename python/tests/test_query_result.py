import pyarrow as pa
from icefalldb.adapter import IcefallDBQueryResult


def test_table_backed_preserves_schema_and_order():
    tbl = pa.table({"id": [1, 2], "name": ["a", "b"]})
    r = IcefallDBQueryResult(tbl)
    assert r.fetchall() == [(1, "a"), (2, "b")]
    assert r.fetchone() == (1, "a")
    assert r.arrow().column_names == ["id", "name"]
    assert r.to_arrow_table().num_rows == 2


def test_empty_result_keeps_schema():
    tbl = pa.table({"id": pa.array([], type=pa.int64())})
    r = IcefallDBQueryResult(tbl)
    assert r.fetchall() == []
    assert r.arrow().schema.field(0).name == "id"


def test_fetchall_map_column_returns_dict():
    """PyArrow to_pylist() returns map columns as list[tuple]; verify we convert to dict."""
    map_type = pa.map_(pa.string(), pa.int64())
    map_array = pa.array([[("a", 1), ("b", 2)], [("c", 3)]], type=map_type)
    tbl = pa.table({"id": [1, 2], "tags": map_array})
    r = IcefallDBQueryResult(tbl)
    rows = r.fetchall()
    assert rows[0] == (1, {"a": 1, "b": 2})
    assert rows[1] == (2, {"c": 3})
    assert r.fetchone() == (1, {"a": 1, "b": 2})


def test_fetchall_list_column_returns_list():
    """List columns should come back as Python lists."""
    list_type = pa.list_(pa.int64())
    list_array = pa.array([[10, 20], [30, 40, 50]], type=list_type)
    tbl = pa.table({"id": [1, 2], "nums": list_array})
    r = IcefallDBQueryResult(tbl)
    rows = r.fetchall()
    assert rows[0] == (1, [10, 20])
    assert rows[1] == (2, [30, 40, 50])


def test_fetchall_struct_column_returns_dict():
    """Struct columns should come back as Python dicts."""
    struct_type = pa.struct([pa.field("x", pa.int64()), pa.field("y", pa.float64())])
    struct_array = pa.array([{"x": 1, "y": 1.5}, {"x": 2, "y": 2.5}], type=struct_type)
    tbl = pa.table({"id": [1, 2], "point": struct_array})
    r = IcefallDBQueryResult(tbl)
    rows = r.fetchall()
    assert rows[0] == (1, {"x": 1, "y": 1.5})
    assert rows[1] == (2, {"x": 2, "y": 2.5})


def test_fetchall_fixed_size_list_column_returns_list():
    """Fixed-size list columns should come back as Python lists."""
    fsl_type = pa.list_(pa.int64(), 3)
    fsl_array = pa.array([[1, 2, 3], [4, 5, 6]], type=fsl_type)
    tbl = pa.table({"id": [1, 2], "vec": fsl_array})
    r = IcefallDBQueryResult(tbl)
    rows = r.fetchall()
    assert rows[0] == (1, [1, 2, 3])
    assert rows[1] == (2, [4, 5, 6])


def test_fetchall_null_complex_values():
    """None values in complex columns should remain None."""
    map_type = pa.map_(pa.string(), pa.int64())
    map_array = pa.array([[("a", 1)], None], type=map_type)
    tbl = pa.table({"id": [1, 2], "tags": map_array})
    r = IcefallDBQueryResult(tbl)
    rows = r.fetchall()
    assert rows[0] == (1, {"a": 1})
    assert rows[1] == (2, None)


def test_fetchall_nested_map_in_list():
    """Nested map-within-list values should be converted recursively."""
    inner_map_type = pa.map_(pa.string(), pa.int64())
    list_of_map_type = pa.list_(inner_map_type)
    # PyArrow accepts dicts for map values when constructing list-of-map arrays.
    maps = pa.array(
        [[{"a": 1}], [{"b": 2}, {"c": 3}]],
        type=list_of_map_type,
    )
    tbl = pa.table({"id": [1, 2], "nested": maps})
    r = IcefallDBQueryResult(tbl)
    rows = r.fetchall()
    assert rows[0] == (1, [{"a": 1}])
    assert rows[1] == (2, [{"b": 2}, {"c": 3}])
