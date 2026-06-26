from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _is_non_decreasing(arr: pa.Array | pa.ChunkedArray) -> bool:
    """Return True if the array values are sorted in non-decreasing order."""
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()

    if arr.null_count > 0:
        arr = arr.drop_null()
    if len(arr) <= 1:
        return True

    if pa.types.is_integer(arr.type):
        # Cast to int64 to avoid overflow in pairwise_diff for narrow types
        # (e.g. int16 [30000, -30000] or uint8 [3, 1]).
        min_diff = pc.min(pc.pairwise_diff(arr.cast(pa.int64()))).as_py()
        return min_diff is not None and min_diff >= 0

    if pa.types.is_timestamp(arr.type):
        try:
            viewed = arr.view(pa.int64())
        except pa.ArrowInvalid:
            return False
        min_diff = pc.min(pc.pairwise_diff(viewed)).as_py()
        return min_diff is not None and min_diff >= 0

    return False


def write_icefalldb_ready_parquet(
    path: str | Path,
    table: pa.Table,
    row_group_size: int = 1_000_000,
    data_page_size: int = 1_048_576,
    sort_keys: list[str] | None = None,
    dictionary_columns: list[str] | None = None,
) -> Path:
    """Write a Parquet file optimized for IcefallDB zero-copy ingest.

    Chooses per-column compression and encoding based on the data shape:

    * Integers and microsecond/nanosecond timestamps: ZSTD level 1, with
      ``DELTA_BINARY_PACKED`` encoding when the column is non-decreasing.
    * Float32/Float64: Snappy compression, ``BYTE_STREAM_SPLIT`` encoding.
    * Utf8/LargeUtf8/Binary/LargeBinary: ZSTD level 1.

    Parameters
    ----------
    path:
        Destination file path.
    table:
        PyArrow Table to write.
    row_group_size:
        Target number of rows per Parquet row group (default 1_000_000).
    data_page_size:
        Target number of bytes per Parquet data page (default 1_048_576).
        Larger pages improve page-index pruning when data is sorted by filter
        columns.
    sort_keys:
        Optional list of column names to sort ascending before writing.
    dictionary_columns:
        Optional list of column names to force dictionary encoding for.
        PyArrow does not expose per-column dictionary control, so passing a
        non-empty list enables dictionary encoding globally.
    """
    path = Path(path)

    schema_names = set(table.schema.names)

    if sort_keys:
        missing = [key for key in sort_keys if key not in schema_names]
        if missing:
            raise ValueError(
                f"sort_keys contain columns not in the table schema: {missing}"
            )
        table = table.sort_by([(key, "ascending") for key in sort_keys])

    if dictionary_columns:
        missing = [key for key in dictionary_columns if key not in schema_names]
        if missing:
            raise ValueError(
                f"dictionary_columns contain columns not in the table schema: {missing}"
            )

    compression: dict[str, str] = {}
    compression_level: dict[str, int] = {}
    column_encoding: dict[str, str] = {}

    for field in table.schema:
        name = field.name
        dtype = field.type

        if pa.types.is_integer(dtype) or pa.types.is_timestamp(dtype):
            compression[name] = "zstd"
            compression_level[name] = 1
            if _is_non_decreasing(table.column(name)):
                column_encoding[name] = "DELTA_BINARY_PACKED"
        elif pa.types.is_floating(dtype):
            compression[name] = "snappy"
            column_encoding[name] = "BYTE_STREAM_SPLIT"
        elif (
            pa.types.is_string(dtype)
            or pa.types.is_large_string(dtype)
            or pa.types.is_binary(dtype)
            or pa.types.is_large_binary(dtype)
        ):
            compression[name] = "zstd"
            compression_level[name] = 1

    # PyArrow supports per-column dictionary control via a list of column names.
    # Keep the existing default (disabled) when dictionary_columns is not
    # supplied, but force dictionary encoding only for the requested columns.
    # Column encoding cannot be combined with dictionary encoding in PyArrow,
    # so omit custom encodings for dictionary-encoded columns.
    dict_cols = set(dictionary_columns) if dictionary_columns else set()
    writer_kwargs: dict[str, object] = {
        "compression": compression,
        "compression_level": compression_level,
        "write_statistics": True,
        "write_page_index": True,
        "data_page_size": data_page_size,
        "store_schema": True,
    }
    if dictionary_columns:
        writer_kwargs["use_dictionary"] = list(dictionary_columns)
        writer_kwargs["column_encoding"] = {
            k: v for k, v in column_encoding.items() if k not in dict_cols
        }
    else:
        writer_kwargs["use_dictionary"] = False
        writer_kwargs["column_encoding"] = column_encoding

    with pq.ParquetWriter(
        path,
        table.schema,
        **writer_kwargs,
    ) as writer:
        for offset in range(0, table.num_rows, row_group_size):
            writer.write_table(table.slice(offset, row_group_size))
    return path
