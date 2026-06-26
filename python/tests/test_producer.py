from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import icefalldb
from icefalldb.producer import _is_non_decreasing


def test_write_icefalldb_ready_parquet_defaults(tmp_path: Path) -> None:
    n = 2_000_000
    table = pa.table(
        {
            "id": range(n),
            "value": [float(i) for i in range(n)],
            "name": [f"name-{i}" for i in range(n)],
        }
    )
    path = icefalldb.write_icefalldb_ready_parquet(
        tmp_path / "test_icefalldb_ready.parquet", table
    )

    meta = pq.read_metadata(path)
    assert meta.num_row_groups == 2
    for rg in range(meta.num_row_groups):
        assert meta.row_group(rg).num_rows == 1_000_000
        id_idx = table.schema.names.index("id")
        value_idx = table.schema.names.index("value")
        name_idx = table.schema.names.index("name")
        assert meta.row_group(rg).column(id_idx).compression == "ZSTD"
        assert meta.row_group(rg).column(value_idx).compression == "SNAPPY"
        assert meta.row_group(rg).column(name_idx).compression == "ZSTD"
        # Sorted id column should use delta encoding.
        assert "DELTA_BINARY_PACKED" in meta.row_group(rg).column(id_idx).encodings

    read = pq.read_table(path)
    assert read.equals(table)


def test_sorted_int64_uses_delta_encoding(tmp_path: Path) -> None:
    table = pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    path = icefalldb.write_icefalldb_ready_parquet(
        tmp_path / "sorted_int64.parquet", table, row_group_size=1024
    )

    meta = pq.read_metadata(path)
    id_idx = table.schema.names.index("id")
    id_col = meta.row_group(0).column(id_idx)
    assert "DELTA_BINARY_PACKED" in id_col.encodings


def test_sorted_int32_uses_delta_encoding(tmp_path: Path) -> None:
    table = pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5], type=pa.int32()),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    path = icefalldb.write_icefalldb_ready_parquet(
        tmp_path / "sorted_int32.parquet", table, row_group_size=1024
    )

    meta = pq.read_metadata(path)
    id_idx = table.schema.names.index("id")
    id_col = meta.row_group(0).column(id_idx)
    assert "DELTA_BINARY_PACKED" in id_col.encodings


def test_unsorted_int_does_not_use_delta_encoding(tmp_path: Path) -> None:
    table = pa.table({"id": pa.array([3, 1, 4, 2, 5], type=pa.int64())})
    path = icefalldb.write_icefalldb_ready_parquet(
        tmp_path / "unsorted_int64.parquet", table, row_group_size=1024
    )

    meta = pq.read_metadata(path)
    id_idx = table.schema.names.index("id")
    id_col = meta.row_group(0).column(id_idx)
    assert "DELTA_BINARY_PACKED" not in id_col.encodings


def test_float64_uses_snappy_compression(tmp_path: Path) -> None:
    table = pa.table({"value": [1.0, 2.0, 3.0]})
    path = icefalldb.write_icefalldb_ready_parquet(
        tmp_path / "float64.parquet", table, row_group_size=1024
    )

    meta = pq.read_metadata(path)
    value_idx = table.schema.names.index("value")
    assert meta.row_group(0).column(value_idx).compression == "SNAPPY"


def test_sort_and_dictionary_columns(tmp_path: Path) -> None:
    table = pa.table(
        {
            "category": ["b", "a", "b", "a"],
            "value": [4, 1, 3, 2],
        }
    )
    path = icefalldb.write_icefalldb_ready_parquet(
        tmp_path / "sorted_dict.parquet",
        table,
        sort_keys=["category"],
        dictionary_columns=["category"],
        row_group_size=1024,
    )

    meta = pq.read_metadata(path)
    assert meta.num_row_groups == 1
    assert meta.num_rows == 4

    category_idx = table.schema.names.index("category")
    category_col = meta.row_group(0).column(category_idx)
    assert (
        "RLE_DICTIONARY" in category_col.encodings
        or "PLAIN_DICTIONARY" in category_col.encodings
    )

    read = pq.read_table(path)
    assert read.column("category").to_pylist() == ["a", "a", "b", "b"]
    assert read.column("value").to_pylist() == [1, 2, 4, 3]


def test_is_non_decreasing_int16_overflow_boundary() -> None:
    arr = pa.array([30000, -30000], type=pa.int16())
    assert _is_non_decreasing(arr) is False


def test_is_non_decreasing_uint8_decreasing() -> None:
    arr = pa.array([3, 1], type=pa.uint8())
    assert _is_non_decreasing(arr) is False


def test_is_non_decreasing_narrow_integer_non_decreasing() -> None:
    arr = pa.array([1, 2, 3, 4, 5], type=pa.int16())
    assert _is_non_decreasing(arr) is True
