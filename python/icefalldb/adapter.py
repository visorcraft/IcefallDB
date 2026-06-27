from __future__ import annotations

import base64
import datetime
import decimal
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


class IcefallDBError(Exception):
    """Raised when a IcefallDB database or table cannot be attached."""


# Reserved names that cannot be used as table names.
_RESERVED_TABLE_NAMES = frozenset(
    {"_manifests", "_schemas", "_staging", "_manifest", "_schema", "_write", "views"}
)

# Pattern for safe identifiers used in SQL and file paths.
_SAFE_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9_\-]+")


def _canonicalize(value):
    """Return a JSON value with object keys sorted recursively."""
    if isinstance(value, dict):
        return {k: _canonicalize(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_canonicalize(v) for v in value]
    return value


def _checksum_json(value: dict) -> str:
    """Compute sha256:<hex> canonical checksum of a JSON object."""
    canonical = _canonicalize(value)
    payload = json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


class _BareNumber(str):
    """Marker for a JSON number that must be serialized without quotes."""


def _preserve_float_text(value, text_value):
    """Return a copy of ``value`` where floats keep their original JSON text."""
    if isinstance(value, float) and isinstance(text_value, str):
        return _BareNumber(text_value)
    if isinstance(value, dict) and isinstance(text_value, dict):
        return {k: _preserve_float_text(v, text_value[k]) for k, v in value.items()}
    if isinstance(value, list) and isinstance(text_value, list):
        return [_preserve_float_text(a, b) for a, b in zip(value, text_value)]
    return value


def _checksum_meta_value(value) -> str:
    """Serialize a JSON value for meta checksum computation.

    Mirrors ``json.dumps(..., separators=(',', ':'))`` but emits
    ``_BareNumber`` values as bare JSON numbers. This preserves the original
    textual representation of floating-point statistics written by the Rust
    core, keeping Rust and Python checksums identical.
    """
    if isinstance(value, dict):
        items = sorted(value.items())
        return (
            "{"
            + ",".join(
                f"{json.dumps(k, ensure_ascii=False, separators=(',', ':'))}:"
                f"{_checksum_meta_value(v)}"
                for k, v in items
            )
            + "}"
        )
    if isinstance(value, list):
        return "[" + ",".join(_checksum_meta_value(v) for v in value) + "]"
    if isinstance(value, _BareNumber):
        return str(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _checksum_meta_bytes(raw: bytes) -> str:
    """Compute sha256:<hex> checksum of a row-group meta file.

    Mirrors the Rust core: canonicalize (sort object keys) and serialize,
    preserving the original textual representation of floating-point numbers.
    """
    text = raw.decode("utf-8")
    value = json.loads(text)
    text_value = json.loads(text, parse_float=str)
    value = _preserve_float_text(value, text_value)
    value["checksum"] = ""
    value["meta_checksum"] = ""
    canonical = _canonicalize(value)
    payload = _checksum_meta_value(canonical).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _checksum_bytes(path: Path) -> str:
    """Compute sha256:<hex> checksum of raw file bytes."""
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
    except PermissionError as exc:
        raise IcefallDBError(f"permission denied reading file: {path}") from exc
    return f"sha256:{hasher.hexdigest()}"


def _read_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise IcefallDBError(f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise IcefallDBError(f"invalid JSON at {path}: {exc}") from exc
    except PermissionError as exc:
        raise IcefallDBError(f"permission denied reading file: {path}") from exc


def _schema_filename(schema_id: int) -> str:
    return f"_schemas/{schema_id:06d}.json"


def _manifest_filename(sequence: int) -> str:
    return f"_manifests/{sequence:09d}.json"


def _current_column_names(schema: dict) -> list[str]:
    dropped = set(schema.get("dropped_columns", []))
    return [c["name"] for c in schema["columns"] if c["name"] not in dropped]


def _icefalldb_type_to_arrow(type_str: str) -> pa.DataType:
    """Convert a IcefallDB primitive type string to a PyArrow DataType."""
    mapping = {
        "int8": pa.int8(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "uint8": pa.uint8(),
        "uint16": pa.uint16(),
        "uint32": pa.uint32(),
        "uint64": pa.uint64(),
        "float32": pa.float32(),
        "float64": pa.float64(),
        "utf8": pa.utf8(),
        "string": pa.utf8(),
        "large_utf8": pa.large_utf8(),
        "binary": pa.binary(),
        "large_binary": pa.large_binary(),
        "bool": pa.bool_(),
        "timestamp": pa.timestamp("us"),
        "timestamp[us]": pa.timestamp("us"),
    }
    base = type_str.strip()
    if base in mapping:
        return mapping[base]
    raise IcefallDBError(
        f"unsupported IcefallDB type for Arrow conversion: {type_str!r}"
    )


def _icefalldb_to_duckdb_type(type_str: str) -> str:
    """Convert a IcefallDB type string to a DuckDB type string.

    Supports primitive types, ``list<T>``, ``large_list<T>``, ``struct<...>``,
    ``map<K,V>``, ``decimal128(p,s)``, and ``fixed_size_binary(n)``.
    """
    return _IcefallDBTypeParser(type_str).parse_type()


class _IcefallDBTypeParser:
    """Recursive-descent parser for IcefallDB type strings."""

    _PRIMITIVE_MAPPING = {
        "int8": "TINYINT",
        "int16": "SMALLINT",
        "int32": "INTEGER",
        "int64": "BIGINT",
        "uint8": "UTINYINT",
        "uint16": "USMALLINT",
        "uint32": "UINTEGER",
        "uint64": "UBIGINT",
        "float32": "FLOAT",
        "float64": "DOUBLE",
        "utf8": "VARCHAR",
        "string": "VARCHAR",
        "large_utf8": "VARCHAR",
        "binary": "BLOB",
        "large_binary": "BLOB",
        "bool": "BOOLEAN",
        "timestamp[us]": "TIMESTAMP",
        "timestamp": "TIMESTAMP",
    }

    def __init__(self, input: str):
        self.input = input
        self.pos = 0
        self.n = len(input)

    def parse_type(self) -> str:
        self.skip_ws()
        word = self.parse_word()
        self.skip_ws()
        if word in self._PRIMITIVE_MAPPING:
            return self._PRIMITIVE_MAPPING[word]
        if word == "list" or word == "large_list":
            self.expect("<")
            inner = self.parse_type()
            self.expect(">")
            return f"{inner}[]"
        if word == "struct":
            self.expect("<")
            fields = self.parse_struct_fields()
            self.expect(">")
            return f"STRUCT({fields})"
        if word == "map":
            self.expect("<")
            key = self.parse_type()
            self.expect(",")
            value = self.parse_type()
            self.expect(">")
            return f"MAP({key}, {value})"
        if word == "decimal128":
            self.expect("(")
            precision = self.parse_number()
            self.expect(",")
            scale = self.parse_number()
            self.expect(")")
            return f"DECIMAL({precision},{scale})"
        if word == "fixed_size_binary":
            self.expect("(")
            self.parse_number()
            self.expect(")")
            return "BLOB"
        raise IcefallDBError(f"unsupported IcefallDB type: {self.input}")

    def parse_struct_fields(self) -> str:
        parts: list[str] = []
        while True:
            self.skip_ws()
            if self.peek() == ">":
                break
            name = self.parse_field_name()
            self.skip_ws()
            self.expect(":")
            field_type = self.parse_type()
            parts.append(f"{name} {field_type}")
            self.skip_ws()
            if self.peek() == ",":
                self.advance()
                continue
            if self.peek() == ">":
                break
            raise IcefallDBError(f"invalid struct type: {self.input}")
        return ", ".join(parts)

    def parse_word(self) -> str:
        self.skip_ws()
        start = self.pos
        c = self.peek()
        if c is None or not (c.isalpha() or c == "_"):
            raise IcefallDBError(f"expected type name in {self.input!r}")
        while self.pos < self.n:
            c = self.input[self.pos]
            if c.isalnum() or c == "_":
                self.pos += 1
            else:
                break
        word = self.input[start : self.pos]
        if word == "timestamp":
            self.skip_ws()
            if self.peek() == "[":
                self.advance()
                bracket_start = self.pos
                while self.pos < self.n and self.input[self.pos] != "]":
                    self.pos += 1
                if self.pos >= self.n:
                    raise IcefallDBError(
                        f"unclosed '[' in timestamp type: {self.input!r}"
                    )
                inner = self.input[bracket_start : self.pos]
                if inner != "us":
                    raise IcefallDBError(
                        f"unsupported timestamp resolution {inner!r} in {self.input!r}"
                    )
                self.advance()  # ']'
        return word

    def parse_field_name(self) -> str:
        self.skip_ws()
        start = self.pos
        c = self.peek()
        if c is None or not (c.isalpha() or c == "_"):
            raise IcefallDBError(f"expected field name in {self.input!r}")
        while self.pos < self.n:
            c = self.input[self.pos]
            if c.isalnum() or c == "_":
                self.pos += 1
            else:
                break
        return self.input[start : self.pos]

    def parse_number(self) -> str:
        self.skip_ws()
        start = self.pos
        while self.pos < self.n and self.input[self.pos].isdigit():
            self.pos += 1
        if start == self.pos:
            raise IcefallDBError(f"expected number in {self.input!r}")
        return self.input[start : self.pos]

    def expect(self, expected: str) -> None:
        self.skip_ws()
        if self.peek() != expected:
            raise IcefallDBError(f"expected {expected!r} in {self.input!r}")
        self.advance()

    def skip_ws(self) -> None:
        while self.pos < self.n and self.input[self.pos].isspace():
            self.pos += 1

    def peek(self) -> str | None:
        if self.pos >= self.n:
            return None
        return self.input[self.pos]

    def advance(self) -> None:
        self.pos += 1


def _validate_table_name(table: str) -> None:
    """Validate that `table` is a safe directory/table name."""
    if not isinstance(table, str):
        raise IcefallDBError(f"table name must be a string, got {type(table).__name__}")
    if not table:
        raise IcefallDBError("table name must not be empty")
    if table.startswith("."):
        raise IcefallDBError(f"table name must not start with '.': {table!r}")
    if "/" in table or "\\" in table or ".." in table:
        raise IcefallDBError(f"invalid table name: {table!r}")
    if table in _RESERVED_TABLE_NAMES:
        raise IcefallDBError(f"reserved table name: {table!r}")
    if not _SAFE_IDENTIFIER_RE.fullmatch(table):
        raise IcefallDBError(f"invalid table name: {table!r}")


def _validate_identifier(name: str, kind: str = "identifier") -> None:
    """Validate an SQL identifier (table or column name)."""
    if not isinstance(name, str):
        raise IcefallDBError(f"{kind} must be a string: {name!r}")
    if '"' in name or "`" in name or "\x00" in name or "\n" in name:
        raise IcefallDBError(f"invalid {kind}: {name!r}")


def _escape_sql_identifier(name: str) -> str:
    """Return a double-quoted SQL identifier, validating it first."""
    _validate_identifier(name, "identifier")
    return f'"{name}"'


def _escape_sql_string_literal(value: str) -> str:
    """Escape a string literal for DuckDB by doubling single quotes."""
    if not isinstance(value, str):
        raise IcefallDBError(f"string literal must be a string: {value!r}")
    return "'" + value.replace("'", "''") + "'"


def _validate_positive_int(value, field: str) -> int:
    """Validate that `value` is a positive integer (not a bool)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise IcefallDBError(f"{field} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise IcefallDBError(f"{field} must be a positive integer, got {value}")
    return value


def _validate_manifest_shape(manifest: dict, table: str, path: Path) -> None:
    """Validate the structure of a manifest object."""
    if not isinstance(manifest, dict):
        raise IcefallDBError(
            f"manifest for table '{table}' is not a JSON object at {path}"
        )
    if "format_version" in manifest:
        version = manifest["format_version"]
        if isinstance(version, bool) or not isinstance(version, int) or version != 1:
            raise IcefallDBError(
                f"unsupported manifest format_version for table '{table}': {version}"
            )
    _validate_positive_int(manifest.get("sequence"), "manifest sequence")
    _validate_positive_int(manifest.get("schema_id"), "manifest schema_id")
    row_groups = manifest.get("row_groups")
    if not isinstance(row_groups, list):
        raise IcefallDBError(
            f"manifest 'row_groups' for table '{table}' must be a list at {path}"
        )
    for idx, rg in enumerate(row_groups):
        if not isinstance(rg, dict):
            raise IcefallDBError(
                f"row group {idx} for table '{table}' is not a JSON object at {path}"
            )
        for key in ("data", "meta"):
            if key not in rg:
                raise IcefallDBError(
                    f"row group {idx} for table '{table}' missing '{key}' at {path}"
                )
            if not isinstance(rg[key], str):
                raise IcefallDBError(
                    f"row group {idx} for table '{table}' has non-string '{key}' "
                    f"at {path}"
                )


def _validate_schema_shape(schema: dict, table: str, path: Path) -> None:
    """Validate the structure of a schema object."""
    if not isinstance(schema, dict):
        raise IcefallDBError(
            f"schema for table '{table}' is not a JSON object at {path}"
        )
    _validate_positive_int(schema.get("schema_id"), "schema schema_id")
    columns = schema.get("columns")
    if not isinstance(columns, list):
        raise IcefallDBError(
            f"schema 'columns' for table '{table}' must be a list at {path}"
        )
    for idx, col in enumerate(columns):
        if not isinstance(col, dict):
            raise IcefallDBError(
                f"column {idx} for table '{table}' is not a JSON object at {path}"
            )
        for key in ("name", "type"):
            if key not in col:
                raise IcefallDBError(
                    f"column {idx} for table '{table}' missing '{key}' at {path}"
                )
            if not isinstance(col[key], str):
                raise IcefallDBError(
                    f"column {idx} for table '{table}' has non-string '{key}' at {path}"
                )
    if "dropped_columns" in schema:
        dropped_columns = schema["dropped_columns"]
        if not isinstance(dropped_columns, list):
            raise IcefallDBError(
                f"schema 'dropped_columns' for table '{table}' must be a list at {path}"
            )
        for idx, entry in enumerate(dropped_columns):
            if not isinstance(entry, str):
                raise IcefallDBError(
                    f"dropped column entry {idx} for table '{table}' "
                    f"must be a string at {path}"
                )


def _resolve_table_path(
    table_dir: Path, rel_path: str, sequence: int, table: str, kind: str
) -> Path:
    """Resolve a relative path inside a table directory and verify it does not escape.

    Rejects empty, absolute, and parent-directory-traversing paths. The
    resolved path is guaranteed to be inside `table_dir`.
    """
    if not rel_path:
        raise IcefallDBError(
            f"empty {kind} path in snapshot {sequence} for table '{table}'"
        )
    if Path(rel_path).is_absolute():
        raise IcefallDBError(
            f"absolute {kind} path in snapshot {sequence} for table '{table}': {rel_path}"
        )
    if ".." in Path(rel_path).parts:
        raise IcefallDBError(
            f"{kind} path escapes table directory in snapshot {sequence} "
            f"for table '{table}': {rel_path}"
        )
    resolved = (table_dir / rel_path).resolve()
    table_resolved = table_dir.resolve()
    try:
        resolved.relative_to(table_resolved)
    except ValueError as exc:
        raise IcefallDBError(
            f"{kind} path escapes table directory in snapshot {sequence} "
            f"for table '{table}': {resolved}"
        ) from exc
    return resolved


def _validate_row_group_meta(
    meta: dict,
    meta_bytes: bytes,
    manifest_schema_id: int,
    expected_data: str,
    sequence: int,
    table: str,
) -> None:
    """Validate the shape and checksum of a row-group meta file."""
    if not isinstance(meta, dict):
        raise IcefallDBError(
            f"row group meta is not a JSON object in snapshot {sequence} "
            f"for table '{table}'"
        )
    _validate_positive_int(meta.get("schema_id"), "row group schema_id")
    if meta["schema_id"] != manifest_schema_id:
        raise IcefallDBError(
            f"row group schema_id mismatch in snapshot {sequence} "
            f"for table '{table}': manifest expects {manifest_schema_id}, "
            f"meta has {meta['schema_id']}"
        )
    meta_data = meta.get("row_group")
    if not isinstance(meta_data, str):
        raise IcefallDBError(
            f"row group meta missing 'row_group' string in snapshot {sequence} "
            f"for table '{table}'"
        )
    # The manifest's `data` path is expected to be the row group identifier with a
    # file extension; the meta file stores the identifier without extension. We
    # require the manifest data path to start with the row group identifier.
    data_path = Path(expected_data)
    if data_path.stem != meta_data:
        raise IcefallDBError(
            f"row group data filename mismatch in snapshot {sequence} "
            f"for table '{table}': manifest has {expected_data!r}, "
            f"meta has row_group {meta_data!r}"
        )

    stored_meta_checksum = meta.get("meta_checksum")
    if not isinstance(stored_meta_checksum, str):
        raise IcefallDBError(
            f"row group meta missing 'meta_checksum' string in snapshot {sequence} "
            f"for table '{table}'"
        )
    # Compute the checksum directly over the raw meta bytes with the checksum
    # fields zeroed. This matches the Rust core and avoids cross-language
    # float-formatting differences in min/max statistics.
    expected_meta_checksum = _checksum_meta_bytes(meta_bytes)
    if stored_meta_checksum != expected_meta_checksum:
        raise IcefallDBError(
            f"row group meta checksum mismatch in snapshot {sequence} "
            f"for table '{table}': expected {expected_meta_checksum}, "
            f"found {stored_meta_checksum}"
        )


class IcefallDBQueryResult:
    """Query result backed by a ``pyarrow.Table`` (schema + column order preserved).

    Provides DuckDB-compatible ``fetchall()`` and ``to_arrow_table()`` methods
    so callers can consume results uniformly regardless of engine.
    """

    def __init__(self, table: pa.Table) -> None:
        if not isinstance(table, pa.Table):
            raise TypeError("IcefallDBQueryResult expects a pyarrow.Table")
        self._table = table

    @classmethod
    def from_rows(cls, rows: list[dict[str, object]]) -> "IcefallDBQueryResult":
        """Construct from a list of dicts (preserves the old construction path)."""
        if not rows:
            return cls(pa.table({}))
        cols = list(rows[0].keys())
        data = {c: [row.get(c) for row in rows] for c in cols}
        return cls(pa.table(data))

    def arrow(self) -> pa.Table:
        """Return the underlying ``pyarrow.Table``."""
        return self._table

    def to_arrow_table(self) -> pa.Table:
        """Return the result as a ``pyarrow.Table``."""
        return self._table

    @staticmethod
    def _convert_arrow_value(value: object, arrow_type: "pa.DataType") -> object:
        """Recursively convert PyArrow-serialised values to DuckDB-compatible Python.

        PyArrow's ``to_pylist()`` returns Arrow ``MapArray`` entries as
        ``list[tuple[key, value]]`` rather than ``dict``.  DuckDB returns
        ``dict``, so we normalise to ``dict`` here so that callers that rely on
        the default engine see a consistent type regardless of which code-path
        (cache vs. live DuckDB relation) produced the result.
        """
        if pa.types.is_map(arrow_type):
            if value is None:
                return None
            # value is list[tuple[k, v]] — convert to dict.
            return {
                IcefallDBQueryResult._convert_arrow_value(k, arrow_type.key_type): (
                    IcefallDBQueryResult._convert_arrow_value(v, arrow_type.item_type)
                )
                for k, v in value
            }
        if (
            pa.types.is_list(arrow_type)
            or pa.types.is_large_list(arrow_type)
            or pa.types.is_fixed_size_list(arrow_type)
        ):
            if value is None:
                return None
            return [
                IcefallDBQueryResult._convert_arrow_value(v, arrow_type.value_type)
                for v in value
            ]
        if pa.types.is_struct(arrow_type):
            if value is None:
                return None
            return {
                field.name: IcefallDBQueryResult._convert_arrow_value(
                    value[field.name], field.type
                )
                for field in arrow_type
            }
        return value

    def _convert_row(self, row: dict, schema: "pa.Schema") -> tuple:
        """Apply ``_convert_arrow_value`` for each column in a result row."""
        return tuple(
            self._convert_arrow_value(row[field.name], field.type) for field in schema
        )

    def fetchall(self) -> list[tuple[object, ...]]:
        """Return all result rows as a list of tuples (column order preserved)."""
        schema = self._table.schema
        if any(
            pa.types.is_map(f.type)
            or pa.types.is_list(f.type)
            or pa.types.is_large_list(f.type)
            or pa.types.is_fixed_size_list(f.type)
            or pa.types.is_struct(f.type)
            for f in schema
        ):
            return [self._convert_row(row, schema) for row in self._table.to_pylist()]
        return [tuple(row.values()) for row in self._table.to_pylist()]

    def fetchone(self) -> tuple[object, ...] | None:
        """Return the first result row as a tuple, or ``None`` if empty."""
        rows = self._table.slice(0, 1).to_pylist()
        if not rows:
            return None
        schema = self._table.schema
        if any(
            pa.types.is_map(f.type)
            or pa.types.is_list(f.type)
            or pa.types.is_large_list(f.type)
            or pa.types.is_fixed_size_list(f.type)
            or pa.types.is_struct(f.type)
            for f in schema
        ):
            return self._convert_row(rows[0], schema)
        return tuple(rows[0].values()) if rows else None

    def df(self):
        """Return the result as a ``pandas.DataFrame``."""
        return self._table.to_pandas()

    def fetchdf(self):
        """Alias for ``df()``."""
        return self._table.to_pandas()


class _NativeIcefallDBConnection:
    """In-process DataFusion connection backed by the Rust PyO3 extension.

    This is the preferred ``engine="datafusion"`` path: it keeps a long-lived
    DataFusion ``SessionContext`` in Rust and returns ``pyarrow.Table`` results
    with zero JSON serialization overhead.
    """

    def __init__(
        self,
        db_path: str | Path,
        tables: list[str],
        snapshot: Optional[int] = None,
        result_cache_mb: Optional[int] = None,
        result_cache_evict: Optional[str] = None,
        key_file: Optional[str | Path] = None,
    ) -> None:
        import icefalldb_query_py

        self.db_path = Path(db_path).resolve()
        self.tables = list(tables)
        self._conn = icefalldb_query_py.IcefallDBConnection(
            str(self.db_path),
            self.tables,
            snapshot=snapshot,
            result_cache_mb=result_cache_mb,
            result_cache_evict=result_cache_evict,
            key_file=str(key_file) if key_file is not None else None,
        )

    def sql(self, sql: str, engine: str | None = None) -> IcefallDBQueryResult:
        """Execute ``sql`` and return a [`IcefallDBQueryResult`]."""
        del engine  # _NativeIcefallDBConnection is DataFusion-only
        table = self._conn.sql(sql)
        return IcefallDBQueryResult(table)

    def mutate(self, sql: str) -> int:
        """Execute a DELETE/UPDATE/MERGE statement and return affected rows.

        Requires a single registered table (the PyO3 bridge resolves the Writer
        from the lone registered table root).
        """
        return self._conn.mutate(sql)

    def clear_cache(self) -> None:
        """Clear the persistent query-result cache for this database."""
        self._conn.clear_cache()


class IcefallDBConnection:
    """Connection handle for the DataFusion query engine.

    Executes SQL by shelling out to the ``icefalldb query`` CLI. This is the
    fallback integration path used when the native PyO3 extension is not
    installed.
    """

    def __init__(
        self,
        db_path: str | Path,
        tables: list[str],
        engine: str = "datafusion",
        key_file: Optional[str | Path] = None,
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self.tables = list(tables)
        self.engine = engine
        self._key_file = key_file

    def sql(self, sql: str, engine: str | None = None) -> IcefallDBQueryResult:
        """Execute ``sql`` and return a [`IcefallDBQueryResult`]."""
        del engine  # IcefallDBConnection is DataFusion-only
        cli = _find_icefalldb_cli()
        if cli is None:
            raise IcefallDBError(
                "icefalldb CLI not found; set the ICEFALLDB_CLI environment variable or "
                "ensure icefalldb is on PATH"
            )

        args = [cli, "query", str(self.db_path)]
        for table in self.tables:
            args.extend(["--table", table])
        if self._key_file is not None:
            args.extend(["--key-file", str(self._key_file)])
        args.append(sql)

        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
            env=_cli_env(),
        )
        if result.returncode != 0:
            raise IcefallDBError(f"icefalldb query failed: {result.stderr}")

        stdout = result.stdout.strip()
        if not stdout:
            return IcefallDBQueryResult.from_rows([])
        try:
            rows = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise IcefallDBError(
                f"icefalldb query returned invalid JSON: {exc}"
            ) from exc
        if not isinstance(rows, list):
            raise IcefallDBError("icefalldb query returned a non-array JSON result")
        return IcefallDBQueryResult.from_rows(rows)


class IcefallDBRouter:
    """Routing connection: DuckDB for SELECTs, native engine for maintenance.

    The icefalldb engine keeps both a DuckDB connection (for fast vectorised
    SELECTs) and a native DataFusion connection (for mutations and as a
    correctness fallback), and routes each statement automatically:

    * ``SELECT`` / read-only statements run on **DuckDB** when every referenced
      table is safe, otherwise on the native engine.
    * ``DELETE`` / ``UPDATE`` / ``MERGE`` run on the **native engine** (which
      owns the Writer + deletion-vector machinery), then the DuckDB views are
      refreshed so subsequent SELECTs see the new snapshot.

    Two safety invariants are enforced before a table is exposed to DuckDB:

    * **Encryption** — a table with a ``_encryption.json`` sidecar is *never*
      routed to DuckDB. DuckDB cannot decrypt IcefallDB's AES-GCM Parquet and,
      with the default plaintext footer, would silently return garbage. Such
      tables always use the native engine.
    * **Deletion vectors** — the DuckDB attach path does not apply IcefallDB
      deletion vectors, so after a DELETE/UPDATE/MERGE a DuckDB view would show
      ghost/duplicate rows. Tables with any ``row_groups[*].deleted_count > 0``
      route to the native engine until compaction removes the tombstones.

    On clean, unencrypted tables (the common benchmark case) every SELECT runs on
    DuckDB at full speed.
    """

    def __init__(
        self,
        db_path: str | Path,
        tables: list[str],
        verify_data_checksums: bool = True,
        connection: Optional[duckdb.DuckDBPyConnection] = None,
        result_cache_mb: Optional[int] = None,
        result_cache_evict: Optional[str] = "lru",
        key_file: Optional[str | Path] = None,
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self.tables = list(tables)
        self.verify_data_checksums = verify_data_checksums
        # Key file for reading encrypted tables (forwarded to native
        # connections). None means keys come from ICEFALLDB_KEY_* env vars.
        self._key_file = key_file

        # Result cache handle: wraps the Rust ResultCacheHandle; None when the
        # native extension is unavailable or caching is not requested.
        if _native_icefalldb_available():
            import icefalldb_query_py

            self._cache = icefalldb_query_py.ResultCacheHandle(
                str(self.db_path),
                list(self.tables),
                result_cache_mb=result_cache_mb,
                result_cache_evict=result_cache_evict,
            )
        else:
            self._cache = None

        # table -> reason ("encrypted" | "dirty") for tables kept off DuckDB.
        self._native_only: dict[str, str] = {}

        self._duck = connection if connection is not None else duckdb.connect()

        # Native connections are built lazily so that a clean-table SELECT
        # workload (the common case) never depends on the native engine, and a
        # failure to load one table does not poison the whole connection. The
        # all-tables connection backs multi-table SELECT fallbacks; per-table
        # connections back mutations.
        self._native_all: Optional[_NativeIcefallDBConnection] = None
        self._native_all_built = False
        self._native_all_error: Optional[BaseException] = None
        self._native_conns: dict[str, _NativeIcefallDBConnection] = {}

        # Per-table manifest-pointer signatures captured at the last refresh,
        # used to detect snapshot advances from concurrent external writers so a
        # stale DuckDB view never serves ghost rows after another process
        # commits a mutation.
        self._manifest_sigs: dict[str, tuple] = {}

        self._refresh_routing(initial=True)

    @staticmethod
    def _manifest_sig(db_path: Path, table: str):
        """Return a cheap change signature (mtime, size, inode) for a table's
        latest-manifest pointer, or None if it cannot be stat'd.

        The mutation WAL is folded into the signature: a deferred DELETE/UPDATE/
        MERGE appends to ``_wal/mutations.log`` without advancing the manifest
        pointer, so a pointer-only signature would miss the change and keep a
        stale native connection / DuckDB view serving the pre-mutation snapshot.
        """
        table_dir = _table_dir_for(db_path, table)
        try:
            ps = (table_dir / "_manifest.json").stat()
            sig = (ps.st_mtime_ns, ps.st_size, ps.st_ino)
        except OSError:
            sig = None
        try:
            ws = (table_dir / "_wal" / "mutations.log").stat()
            wal = (ws.st_mtime_ns, ws.st_size, ws.st_ino)
        except OSError:
            wal = None
        if sig is None and wal is None:
            return None
        return (sig, wal)

    def _ensure_fresh(self, sql: str) -> None:
        """Re-evaluate routing if any referenced table's snapshot advanced.

        IcefallDB allows a single writer per table (flock); another process may
        commit a mutation (e.g. a DELETE introducing deletion vectors) after this
        connection classified its tables. Without this check the cached DuckDB
        view would keep serving the pre-mutation snapshot, violating the
        deletion-vector guard. The check is one ``stat()`` per referenced table.
        """
        for table in self.tables:
            if _table_referenced_in_sql(sql, table):
                if self._manifest_sig(self.db_path, table) != self._manifest_sigs.get(
                    table
                ):
                    self._refresh_routing()
                    return

    # -- native connection plumbing ----------------------------------------
    def _get_native_all(self) -> _NativeIcefallDBConnection:
        """Return the all-tables native connection, building it on first use.

        Construction errors are captured and re-raised every time native is
        actually required, rather than swallowed at attach time.
        """
        if not self._native_all_built:
            self._native_all_built = True
            self._native_all = None
            self._native_all_error = None
            if not _native_icefalldb_available():
                self._native_all_error = IcefallDBError(
                    "the native DataFusion extension (icefalldb_query_py) is not "
                    "available; install/rebuild it for mutations or encrypted/"
                    "post-mutation reads"
                )
            else:
                try:
                    self._native_all = _NativeIcefallDBConnection(
                        self.db_path, self.tables, key_file=self._key_file
                    )
                except Exception as exc:  # surfaced via the raise below
                    self._native_all_error = exc
        if self._native_all is None:
            raise IcefallDBError(
                "the native DataFusion engine could not be initialised"
            ) from self._native_all_error
        return self._native_all

    def _invalidate_native_all(self) -> None:
        """Force the all-tables native connection to be rebuilt on next use."""
        self._native_all = None
        self._native_all_built = False
        self._native_all_error = None

    # -- routing state -----------------------------------------------------
    def _classify_table(self, table: str) -> str:
        if _table_is_encrypted(self.db_path, table):
            return "encrypted"
        if _table_has_active_deletions(self.db_path, table):
            return "dirty"
        return "duckdb"

    def _refresh_routing(self, initial: bool = False) -> None:
        """Re-evaluate per-table routing and rebuild DuckDB views for safe tables.

        Also refreshes the manifest-pointer signatures used by ``_ensure_fresh``
        to detect concurrent external writes between calls.
        """
        verify = self.verify_data_checksums if initial else False
        for table in self.tables:
            new_sig = self._manifest_sig(self.db_path, table)
            if not initial and new_sig != self._manifest_sigs.get(table):
                # The table's snapshot advanced (an external writer committed, or
                # another table's mutation triggered this refresh). Cached native
                # connections are pinned to their construction snapshot and the
                # metadata-aggregate fast path reads from that pinned manifest
                # without a scan, so a stale connection would keep serving the
                # pre-advance row counts (and, via that stale read, poison the
                # shared result cache for later connections). Drop them so the
                # next native query rebuilds against the new snapshot.
                self._native_conns.pop(table, None)
                self._invalidate_native_all()
            self._manifest_sigs[table] = new_sig
            kind = self._classify_table(table)
            if kind == "duckdb":
                _attach_table(
                    self._duck,
                    self.db_path,
                    table,
                    verify_data_checksums=verify,
                )
                self._native_only.pop(table, None)
            else:
                # Drop any stale view so DuckDB can never read this table.
                self._duck.execute(
                    f"DROP VIEW IF EXISTS {_escape_sql_identifier(table)}"
                )
                self._native_only[table] = kind

    def _duckdb_safe(self, sql: str) -> bool:
        """True if every attached table referenced in ``sql`` is DuckDB-safe."""
        referenced = [t for t in self.tables if _table_referenced_in_sql(sql, t)]
        if not referenced:
            # No known table referenced (e.g. a SELECT over DuckDB-only schema
            # functions). DuckDB handles it directly.
            return True
        return all(t not in self._native_only for t in referenced)

    # -- SQL entry point ---------------------------------------------------
    def sql(self, sql: str, engine: str | None = None) -> Any:
        """Execute ``sql``, routing between DuckDB and the native engine.

        Returns a DuckDB relation (for SELECTs on DuckDB) or a
        [`IcefallDBQueryResult`] (for native results / mutations), both of which
        expose ``fetchall()`` for uniform consumption.
        """
        del engine  # the hybrid engine selects the backend per statement
        kind = _classify_sql(sql)
        if kind in ("delete", "update", "merge"):
            affected = self._mutate(sql)
            return IcefallDBQueryResult.from_rows([{"affected_rows": affected}])
        if kind == "select":
            # Detect concurrent external writes before consulting cached views
            # so a stale DuckDB view can never serve ghost rows.
            self._ensure_fresh(sql)
            if self._metadata_target(sql) is not None:
                return self._run_metadata_or_duckdb(sql)
            if self._duckdb_safe(sql):
                if self._cache is not None:
                    hit = self._cache.get(sql)  # None on miss or ineligible
                    if hit is not None:
                        return IcefallDBQueryResult(hit)
                    tbl = self._duck.sql(sql).to_arrow_table()
                    self._cache.put(sql, tbl)
                    return IcefallDBQueryResult(tbl)
                return self._duck.sql(sql)
        # Encrypted / dirty-table SELECT, INSERT, or other statements: native.
        return self._get_native_all().sql(sql)

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        """Alias for :meth:`sql` — routes per-statement exactly like ``sql()``.

        Provided so that ``IcefallDBRouter`` is a drop-in for callers that use
        the DuckDB ``connection.execute(sql).fetchall()`` pattern.
        """
        if args or kwargs:
            raise TypeError(
                "IcefallDBRouter.execute() does not support query parameters; "
                "inline literals or use the native engine"
            )
        return self.sql(sql)

    # -- metadata-aggregate fast path --------------------------------------
    def _metadata_target(self, sql: str) -> Optional[str]:
        """Return the table for a metadata-aggregate query on a DuckDB-safe table.

        Only fires for clean (non-encrypted, no deletion vectors) tables, since
        those already route to native via the safety guards. Dirty/encrypted
        tables are handled by the required-native path below.
        """
        target = _metadata_aggregate_target(sql)
        if target is None or target not in self.tables or target in self._native_only:
            return None
        return target

    def _run_metadata_or_duckdb(self, sql: str) -> Any:
        """Run an unfiltered aggregate on a single-table native connection.

        Native is an optimisation here (DuckDB would also answer correctly), so
        any failure to build/serve native falls back to DuckDB rather than
        surfacing an error. This keeps a load failure on some other attached
        table from breaking COUNT(*) on a clean one.
        """
        target = _metadata_aggregate_target(sql)
        assert target is not None  # checked by _metadata_target before dispatch
        try:
            if not _native_icefalldb_available():
                raise IcefallDBError("native extension unavailable")
            return self._single_table_native(target).sql(sql)
        except Exception:
            return self._duck.sql(sql)

    # -- mutations ---------------------------------------------------------
    def mutate(self, sql: str) -> int:
        """Execute a DELETE/UPDATE/MERGE on the native engine; return affected rows."""
        return self._mutate(sql)

    def _mutate(self, sql: str) -> int:
        target = _extract_mutation_table(sql)
        if target is None and len(self.tables) == 1:
            target = self.tables[0]
        if target is None:
            raise IcefallDBError(
                "could not determine the target table for the mutation; attach a "
                "single table or use an unqualified DELETE FROM <table> / "
                "UPDATE <table> / MERGE INTO <table> statement"
            )
        if target not in self.tables:
            raise IcefallDBError(
                f"mutation target table {target!r} is not attached to this connection"
            )
        # Mutations run on a single-table native connection, so a statement that
        # references any *other* attached table cannot be served (the other
        # table is not registered there). Fail early with a clear message rather
        # than a confusing "table not found" from deep in the native engine.
        others = [
            t for t in self.tables if t != target and _table_referenced_in_sql(sql, t)
        ]
        if others:
            raise IcefallDBError(
                f"icefalldb mutations support a single target table, but this "
                f"statement also references {others!r}; use the native engine "
                f"directly for cross-table MERGE/UPDATE/DELETE"
            )
        conn = self._single_table_native(target)
        affected = conn.mutate(sql)
        # `conn` applied its own delta, so it is already current for `target`.
        # Record the new signature before refreshing so _refresh_routing's
        # snapshot-change check does not drop this freshly-current connection as
        # "stale" (which would force a costly rebuild on every mutation).
        self._manifest_sigs[target] = self._manifest_sig(self.db_path, target)
        # Invalidate the all-tables native connection so a subsequent SELECT
        # fallback sees the new snapshot, and re-evaluate routing (the mutation
        # may have introduced deletion vectors).
        self._invalidate_native_all()
        self._refresh_routing()
        # No result-cache wipe. The mutation made `target` "dirty" (a non-empty
        # WAL log, or new deletion vectors), so `_refresh_routing` above
        # reclassified it onto the native engine: any later SELECT touching
        # `target` runs `_ensure_fresh` and routes to native, never consulting
        # the DuckDB result cache, so its pre-mutation entries cannot be served.
        # Cached results for *other* tables stay valid, and the reference-aware
        # cache key keeps them valid even once `target`'s manifest later folds.
        return affected

    def _single_table_native(self, table: str) -> _NativeIcefallDBConnection:
        """Return a cached single-table native connection for ``table``.

        Single-table connections isolate load failures: a problem with one
        attached table cannot prevent serving another. Used for both mutations
        and the metadata-aggregate fast path.
        """
        if table not in self._native_conns:
            if not _native_icefalldb_available():
                raise IcefallDBError(
                    "the native DataFusion extension (icefalldb_query_py) is "
                    "required for this operation and is not available"
                )
            self._native_conns[table] = _NativeIcefallDBConnection(
                self.db_path, [table], key_file=self._key_file
            )
        return self._native_conns[table]

    # -- misc --------------------------------------------------------------
    def clear_cache(self) -> None:
        """Clear the native engine's persistent query-result cache."""
        if self._native_all is not None:
            self._native_all.clear_cache()
        for conn in self._native_conns.values():
            conn.clear_cache()

    def close(self) -> None:
        """Close the DuckDB connection and release native resources."""
        self._duck.close()
        self._native_all = None
        self._native_all_built = False
        self._native_conns.clear()


def _native_icefalldb_available() -> bool:
    """Return True if the native PyO3 DataFusion extension is importable."""
    try:
        import icefalldb_query_py  # noqa: F401

        return True
    except Exception:
        return False


def _attach_table(
    connection: duckdb.DuckDBPyConnection,
    db_path: Path,
    table: str,
    verify_data_checksums: bool = True,
) -> None:
    _validate_table_name(table)

    table_dir = db_path / table
    if not table_dir.is_dir():
        view_dir = db_path / "views" / table
        if view_dir.is_dir():
            table_dir = view_dir
    manifest_pointer_path = table_dir / "_manifest.json"
    if not manifest_pointer_path.is_file():
        raise IcefallDBError(
            f"table '{table}' has no manifest pointer at {manifest_pointer_path}"
        )

    pointer = _read_json(manifest_pointer_path)
    if not isinstance(pointer, dict):
        raise IcefallDBError(
            f"manifest pointer for table '{table}' is not a JSON object at "
            f"{manifest_pointer_path}"
        )
    latest_raw = pointer.get("latest")
    if isinstance(latest_raw, bool) or not isinstance(latest_raw, int):
        raise IcefallDBError(
            f"invalid manifest pointer for table '{table}' at "
            f"{manifest_pointer_path}: latest must be an integer"
        )
    sequence = latest_raw
    if sequence < 0:
        raise IcefallDBError(
            f"invalid manifest pointer for table '{table}' at "
            f"{manifest_pointer_path}: latest must be non-negative"
        )

    # An empty table (latest == 0) is valid: register a zero-row relation with
    # the current schema columns.
    if sequence == 0:
        schema_pointer_path = table_dir / "_schema.json"
        if not schema_pointer_path.is_file():
            raise IcefallDBError(
                f"empty table '{table}' has no schema pointer at {schema_pointer_path}"
            )
        schema_pointer = _read_json(schema_pointer_path)
        if not isinstance(schema_pointer, dict):
            raise IcefallDBError(
                f"schema pointer for table '{table}' is not a JSON object"
            )
        schema_id = _validate_positive_int(
            schema_pointer.get("latest"), "schema pointer latest"
        )
        schema_path = table_dir / _schema_filename(schema_id)
        schema = _read_json(schema_path)
        _validate_schema_shape(schema, table, schema_path)
        columns = _current_column_names(schema)
        if not columns:
            raise IcefallDBError(f"table '{table}' has no queryable columns")
        select_list = ", ".join(_escape_sql_identifier(c) for c in columns)
        values = ", ".join(
            f"CAST(NULL AS {_icefalldb_to_duckdb_type(col['type'])})"
            for col in schema["columns"]
            if col["name"] in columns
        )
        table_ident = _escape_sql_identifier(table)
        sql = (
            f"CREATE OR REPLACE VIEW {table_ident} AS "
            f"SELECT {select_list} FROM (VALUES ({values})) AS t({select_list}) LIMIT 0"
        )
        connection.execute(sql)
        return

    manifest_path = table_dir / _manifest_filename(sequence)
    manifest = _read_json(manifest_path)
    _validate_manifest_shape(manifest, table, manifest_path)
    if manifest["sequence"] != sequence:
        raise IcefallDBError(
            f"manifest sequence mismatch for table '{table}': "
            f"pointer expects {sequence}, manifest has {manifest['sequence']}"
        )

    stored_checksum = manifest.get("checksum")
    if not isinstance(stored_checksum, str):
        raise IcefallDBError(
            f"manifest checksum missing for table '{table}' at {manifest_path}"
        )
    manifest_for_checksum = dict(manifest)
    manifest_for_checksum["checksum"] = ""
    expected_checksum = _checksum_json(manifest_for_checksum)
    if stored_checksum != expected_checksum:
        raise IcefallDBError(
            f"manifest checksum mismatch for table '{table}' at {manifest_path}: "
            f"expected {expected_checksum}, found {stored_checksum}"
        )

    schema_id = manifest["schema_id"]
    schema_path = table_dir / _schema_filename(schema_id)
    schema = _read_json(schema_path)
    _validate_schema_shape(schema, table, schema_path)
    if schema["schema_id"] != schema_id:
        raise IcefallDBError(
            f"schema id mismatch for table '{table}': "
            f"manifest expects {schema_id}, schema has {schema['schema_id']}"
        )

    columns = _current_column_names(schema)
    if not columns:
        raise IcefallDBError(f"table '{table}' has no queryable columns")
    select_list = ", ".join(_escape_sql_identifier(c) for c in columns)

    data_files = []
    for rg in manifest["row_groups"]:
        data_path = _resolve_table_path(
            table_dir, rg["data"], sequence, table, "row group data"
        )
        if not data_path.is_file():
            raise IcefallDBError(
                f"missing row group file in snapshot {sequence} for table '{table}': {data_path}"
            )
        meta_path = _resolve_table_path(
            table_dir, rg["meta"], sequence, table, "row group meta"
        )
        meta = _read_json(meta_path)
        meta_bytes = meta_path.read_bytes()
        _validate_row_group_meta(
            meta,
            meta_bytes,
            schema_id,
            rg["data"],
            sequence,
            table,
        )
        stored_checksum = meta.get("checksum")
        if not isinstance(stored_checksum, str):
            raise IcefallDBError(
                f"row group meta missing 'checksum' string in snapshot {sequence} "
                f"for table '{table}'"
            )
        if verify_data_checksums:
            actual_checksum = _checksum_bytes(data_path)
            if actual_checksum != stored_checksum:
                raise IcefallDBError(
                    f"row group data checksum mismatch in snapshot {sequence} "
                    f"for table '{table}': expected {stored_checksum}, "
                    f"found {actual_checksum} for {data_path}"
                )
        data_files.append(str(data_path))

    if data_files:
        files_literal = ", ".join(_escape_sql_string_literal(f) for f in data_files)
        table_ident = _escape_sql_identifier(table)
        dropped_columns = set(schema.get("dropped_columns", []))
        if dropped_columns:
            # Add a zero-row dummy relation with the current schema so that
            # columns missing from every Parquet file are still present as NULL.
            dummy_values = ", ".join(
                f"CAST(NULL AS {_icefalldb_to_duckdb_type(col['type'])}) AS {_escape_sql_identifier(col['name'])}"
                for col in schema["columns"]
                if col["name"] in columns
            )
            sql = (
                f"CREATE OR REPLACE VIEW {table_ident} AS "
                f"SELECT {select_list} FROM ("
                f"SELECT * FROM read_parquet([{files_literal}], union_by_name=True, filename=False) "
                f"UNION ALL BY NAME "
                f"SELECT {dummy_values} WHERE FALSE"
                f") AS u"
            )
        else:
            sql = (
                f"CREATE OR REPLACE VIEW {table_ident} AS "
                f"SELECT {select_list} FROM read_parquet([{files_literal}], union_by_name=True, filename=False)"
            )
        connection.execute(sql)
    else:
        # Empty table: return a zero-row relation with the correct column names and types.
        values = ", ".join(
            f"CAST(NULL AS {_icefalldb_to_duckdb_type(col['type'])})"
            for col in schema["columns"]
            if col["name"] in columns
        )
        table_ident = _escape_sql_identifier(table)
        sql = (
            f"CREATE OR REPLACE VIEW {table_ident} AS "
            f"SELECT {select_list} FROM (VALUES ({values})) AS t({select_list}) LIMIT 0"
        )
        connection.execute(sql)


def _table_dir_for(db_path: Path, table: str) -> Path:
    """Resolve the directory holding ``table``'s manifest (table or derived view)."""
    table_dir = db_path / table
    if not table_dir.is_dir():
        view_dir = db_path / "views" / table
        if view_dir.is_dir():
            table_dir = view_dir
    return table_dir


def _table_is_encrypted(db_path: Path, table: str) -> bool:
    """Return True if ``table`` has a Parquet modular-encryption marker.

    Encryption is signalled solely by a ``_encryption.json`` sidecar; it is not
    recorded in the manifest or schema. DuckDB cannot decrypt these files, so the
    hybrid router must keep encrypted tables off the DuckDB path.
    """
    return _table_dir_for(db_path, table).joinpath("_encryption.json").is_file()


def _read_latest_manifest(db_path: Path, table: str) -> Optional[dict]:
    """Return the latest committed manifest for ``table``, or None if empty/missing."""
    table_dir = _table_dir_for(db_path, table)
    pointer_path = table_dir / "_manifest.json"
    if not pointer_path.is_file():
        return None
    pointer = _read_json(pointer_path)
    latest = pointer.get("latest")
    if not isinstance(latest, int) or isinstance(latest, bool) or latest <= 0:
        return None
    manifest_path = table_dir / _manifest_filename(latest)
    if not manifest_path.is_file():
        return None
    return _read_json(manifest_path)


def _table_has_active_deletions(db_path: Path, table: str) -> bool:
    """Return True if ``table`` has any tombstoned rows in its latest snapshot.

    The DuckDB attach path does not apply IcefallDB deletion vectors, so a table
    with active deletions would expose ghost/duplicate rows through DuckDB. Such
    tables must route to the native engine until compaction rewrites them away.

    A pending mutation WAL also routes to native: its deferred DELETE/UPDATE/MERGE
    records are not yet reflected in the checkpoint manifest the DuckDB path
    reads, so a DuckDB view would be stale until the next checkpoint.
    """
    wal_log = _table_dir_for(db_path, table) / "_wal" / "mutations.log"
    try:
        if wal_log.is_file() and wal_log.stat().st_size > 0:
            return True
    except OSError:
        pass
    manifest = _read_latest_manifest(db_path, table)
    if manifest is None:
        return False
    for rg in manifest.get("row_groups", []):
        if rg.get("deleted_count", 0) > 0:
            return True
    return False


def read_arrow_table(
    db_path: str | Path,
    table: Optional[str] = None,
    columns: Optional[list[str]] = None,
    verify_data_checksums: bool = False,
) -> pa.Table:
    """Read a IcefallDB table directly into a PyArrow Table.

    This bypasses DuckDB and reads the underlying row-group Parquet files
    directly, so it is much faster for full-table scans than executing
    ``SELECT *`` through the SQL view.

    Parameters
    ----------
    db_path:
        Path to the IcefallDB database directory, or to a single table directory.
    table:
        Table name. Required when ``db_path`` is a database directory; ignored
        when ``db_path`` points directly at a table directory.
    columns:
        Optional list of column names to read. If omitted, all current columns
        are returned.
    verify_data_checksums:
        If ``True``, verify the SHA-256 checksum of every row-group data file
        before reading it.

    Returns
    -------
    pa.Table
        The table contents as a PyArrow Table.
    """
    db_path = Path(db_path).resolve()
    if not db_path.is_dir():
        raise IcefallDBError(f"database directory not found: {db_path}")

    # Allow callers to pass either a database directory containing table
    # subdirectories, or a single table directory directly.
    if (db_path / "_manifest.json").is_file():
        table_dir = db_path
        db_path = table_dir.parent
        if table is None:
            table = table_dir.name
    else:
        if table is None:
            raise IcefallDBError(
                "table name is required when db_path is a database directory"
            )
        table_dir = db_path / table

    _validate_table_name(table)
    if not table_dir.is_dir():
        raise IcefallDBError(f"table directory not found: {table_dir}")

    manifest_pointer_path = table_dir / "_manifest.json"
    if not manifest_pointer_path.is_file():
        raise IcefallDBError(
            f"table '{table}' has no manifest pointer at {manifest_pointer_path}"
        )
    pointer = _read_json(manifest_pointer_path)
    sequence = _validate_positive_int(pointer.get("latest"), "manifest pointer latest")

    # Empty table: return a zero-row relation with the current schema.
    if sequence == 0:
        schema_pointer_path = table_dir / "_schema.json"
        if not schema_pointer_path.is_file():
            raise IcefallDBError(
                f"empty table '{table}' has no schema pointer at {schema_pointer_path}"
            )
        schema_pointer = _read_json(schema_pointer_path)
        schema_id = _validate_positive_int(
            schema_pointer.get("latest"), "schema pointer latest"
        )
        schema_path = table_dir / _schema_filename(schema_id)
        schema = _read_json(schema_path)
        _validate_schema_shape(schema, table, schema_path)
        arrow_fields = [
            pa.field(
                col["name"],
                _icefalldb_type_to_arrow(col["type"]),
                nullable=col.get("nullable", True),
            )
            for col in schema["columns"]
            if col["name"] not in set(schema.get("dropped_columns", []))
        ]
        return pa.Table.from_arrays(
            [pa.nulls(0, field.type) for field in arrow_fields],
            schema=pa.schema(arrow_fields),
        )

    manifest_path = table_dir / _manifest_filename(sequence)
    manifest = _read_json(manifest_path)
    _validate_manifest_shape(manifest, table, manifest_path)
    if manifest["sequence"] != sequence:
        raise IcefallDBError(
            f"manifest sequence mismatch for table '{table}': "
            f"pointer expects {sequence}, manifest has {manifest['sequence']}"
        )

    schema_id = manifest["schema_id"]
    schema_path = table_dir / _schema_filename(schema_id)
    schema = _read_json(schema_path)
    _validate_schema_shape(schema, table, schema_path)
    if schema["schema_id"] != schema_id:
        raise IcefallDBError(
            f"schema id mismatch for table '{table}': "
            f"manifest expects {schema_id}, schema has {schema['schema_id']}"
        )

    current_columns = _current_column_names(schema)
    if not current_columns:
        raise IcefallDBError(f"table '{table}' has no queryable columns")

    data_files: list[str] = []
    for rg in manifest["row_groups"]:
        data_path = _resolve_table_path(
            table_dir, rg["data"], sequence, table, "row group data"
        )
        if not data_path.is_file():
            raise IcefallDBError(
                f"missing row group file in snapshot {sequence} for table '{table}': {data_path}"
            )
        if verify_data_checksums:
            meta_path = _resolve_table_path(
                table_dir, rg["meta"], sequence, table, "row group meta"
            )
            meta = _read_json(meta_path)
            _validate_row_group_meta(meta, schema_id, rg["data"], sequence, table)
            stored_checksum = meta.get("checksum")
            if not isinstance(stored_checksum, str):
                raise IcefallDBError(
                    f"row group meta missing 'checksum' string in snapshot {sequence} "
                    f"for table '{table}'"
                )
            actual_checksum = _checksum_bytes(data_path)
            if actual_checksum != stored_checksum:
                raise IcefallDBError(
                    f"row group data checksum mismatch in snapshot {sequence} "
                    f"for table '{table}': expected {stored_checksum}, "
                    f"found {actual_checksum} for {data_path}"
                )
        data_files.append(str(data_path))

    if not data_files:
        arrow_fields = [
            pa.field(
                col["name"],
                _icefalldb_type_to_arrow(col["type"]),
                nullable=col.get("nullable", True),
            )
            for col in schema["columns"]
            if col["name"] in current_columns
        ]
        return pa.Table.from_arrays(
            [pa.nulls(0, field.type) for field in arrow_fields],
            schema=pa.schema(arrow_fields),
        )

    read_columns = list(columns) if columns is not None else current_columns
    invalid = [c for c in read_columns if c not in current_columns]
    if invalid:
        raise IcefallDBError(f"requested columns not in table '{table}': {invalid}")

    table_obj = pq.read_table(
        data_files, columns=read_columns, memory_map=True, pre_buffer=True
    )

    # Add null columns for any current columns that are missing from the
    # underlying Parquet files (e.g. dropped columns or schema evolution).
    result_names = table_obj.column_names
    missing = [c for c in current_columns if c not in result_names]
    if missing:
        arrays = [table_obj.column(name) for name in result_names]
        names = list(result_names)
        nrows = table_obj.num_rows
        for col_name in missing:
            col = next(col for col in schema["columns"] if col["name"] == col_name)
            arrays.append(pa.nulls(nrows, _icefalldb_type_to_arrow(col["type"])))
            names.append(col_name)
        ordered_arrays = [arrays[names.index(c)] for c in current_columns]
        table_obj = pa.Table.from_arrays(ordered_arrays, names=current_columns)

    return table_obj


def _discover_tables(db_path: Path) -> list[str]:
    try:
        tables = sorted(
            p.name
            for p in db_path.iterdir()
            if p.is_dir() and (p / "_manifest.json").is_file()
        )
        views_dir = db_path / "views"
        if views_dir.is_dir():
            views = sorted(
                p.name
                for p in views_dir.iterdir()
                if p.is_dir() and (p / "_manifest.json").is_file()
            )
            # Views are registered under their bare name; skip any name that
            # conflicts with a top-level table.
            for view_name in views:
                if view_name not in tables:
                    tables.append(view_name)
        return tables
    except PermissionError as exc:
        raise IcefallDBError(f"cannot read database directory: {db_path}") from exc


def attach(
    db_path: str | Path,
    connection: Optional[duckdb.DuckDBPyConnection] = None,
    tables: Optional[Iterable[str]] = None,
    verify_data_checksums: bool = True,
    engine: str = "icefalldb",
    result_cache_mb: Optional[int] = None,
    result_cache_evict: Optional[str] = "lru",
    snapshot: Optional[int] = None,
    key_file: Optional[str | Path] = None,
) -> duckdb.DuckDBPyConnection | IcefallDBConnection | IcefallDBRouter:
    """Attach IcefallDB tables from `db_path` to a query engine.

    If `connection` is omitted, a new in-memory DuckDB connection is created for
    the DuckDB engine.
    If `tables` is omitted, every subdirectory containing `_manifest.json` is
    attached as a view named after the directory.

    Set ``engine="datafusion"`` to use the native DataFusion query engine via the
    ``icefalldb query`` CLI. In that case a [`IcefallDBConnection`] is returned instead
    of a DuckDB connection.

    Set ``engine="icefalldb"`` to use a routing connection that runs SELECTs on
    DuckDB (for fast vectorised scans) and mutations on the native DataFusion
    engine. In that case a [`IcefallDBRouter`] is returned. Encrypted tables and
    tables with active deletion vectors automatically fall back to the native
    engine for correctness.

    Set `verify_data_checksums=False` to skip the per-row-group SHA-256 data
    checksum verification on attach. This speeds up read-heavy workloads where
    the caller trusts the stored metadata checksums; corruption can still be
    detected offline with ``icefalldb check``.

    Set ``snapshot=N`` to open a read-only connection pinned to historical
    snapshot *N*. This always returns a read-only NATIVE connection
    (``IcefallDBConnection``) regardless of ``engine=`` — ``engine="icefalldb"`` and
    ``engine="datafusion"`` both route to native for time travel, and
    ``engine="duckdb"`` raises ``ValueError`` (DuckDB cannot read deletion
    vectors / time-travel). Mutations on a snapshot-pinned connection raise an
    error.
    """
    if engine == "hybrid":  # renamed to "icefalldb"
        raise ValueError("engine 'hybrid' was renamed to 'icefalldb'")
    if engine not in ("duckdb", "datafusion", "icefalldb"):
        raise ValueError(f"unsupported engine: {engine!r}")
    if snapshot is not None and engine == "duckdb":
        raise ValueError(
            "engine='duckdb' cannot time-travel; snapshot reads require the native engine"
        )

    db_path = Path(db_path).resolve()
    if not db_path.is_dir():
        raise IcefallDBError(f"database directory not found: {db_path}")

    # Allow callers to pass either a database directory containing table
    # subdirectories, or a single table directory directly.
    single_table_dir = None
    if (db_path / "_manifest.json").is_file():
        single_table_dir = db_path
        db_path = single_table_dir.parent
        if tables is None:
            tables = [single_table_dir.name]

    if tables is not None:
        if isinstance(tables, str):
            tables = [tables]
        table_names = list(tables)
    else:
        table_names = _discover_tables(db_path)

    # Validate table names before dispatching to any engine so that callers
    # always get a IcefallDBError (not an engine-specific exception) on bad names.
    for _tname in table_names:
        _validate_table_name(_tname)

    # DuckDB cannot decrypt Parquet Modular Encryption; refuse rather than build
    # read_parquet views that would fail or return ciphertext.
    if engine == "duckdb":
        for _tname in table_names:
            if _table_is_encrypted(db_path, _tname):
                raise IcefallDBError(
                    f"table '{_tname}' is encrypted; engine='duckdb' cannot read it. "
                    "Use the default engine or engine='datafusion' with the key "
                    "(ICEFALLDB_KEY_* env vars or key_file=)."
                )

    if engine == "datafusion" or snapshot is not None:
        if _native_icefalldb_available():
            return _NativeIcefallDBConnection(
                db_path,
                table_names,
                snapshot=snapshot,
                result_cache_mb=result_cache_mb,
                result_cache_evict=result_cache_evict,
                key_file=key_file,
            )
        if snapshot is not None:
            raise IcefallDBError(
                "the native DataFusion extension (icefalldb_query_py) is required for "
                "snapshot time-travel reads; install/rebuild it"
            )
        return IcefallDBConnection(
            db_path, table_names, engine="datafusion", key_file=key_file
        )

    if engine == "icefalldb":
        return IcefallDBRouter(
            db_path,
            table_names,
            verify_data_checksums=verify_data_checksums,
            connection=connection,
            result_cache_mb=result_cache_mb,
            result_cache_evict=result_cache_evict,
            key_file=key_file,
        )

    con = connection or duckdb.connect()
    for table in table_names:
        _attach_table(con, db_path, table, verify_data_checksums=verify_data_checksums)

    return con


def snapshots(db_path: str | Path, table: str) -> list[dict]:
    """Return all committed snapshots for ``table`` in ``db_path``.

    Requires the native DataFusion extension (``icefalldb_query_py``).

    Each entry is a dict with keys:
    - ``sequence`` (int): snapshot sequence number.
    - ``committed_at`` (str or None): ISO-8601 commit timestamp.
    - ``rows`` (int): total row count in the snapshot.
    - ``fragments`` (int): number of row-group fragments.
    - ``parent_hash`` (str or None): manifest chain hash of the parent snapshot.

    Snapshots are returned sorted ascending by sequence number.
    """
    import icefalldb_query_py

    return icefalldb_query_py.snapshots(str(Path(db_path).resolve()), table)


def attach_table(
    db_path: str | Path,
    table: str,
    connection: Optional[duckdb.DuckDBPyConnection] = None,
    verify_data_checksums: bool = True,
    engine: str = "icefalldb",
) -> duckdb.DuckDBPyConnection | IcefallDBConnection | IcefallDBRouter:
    """Attach a single IcefallDB table from `db_path` to a query engine.

    Set ``engine="datafusion"`` to use the native DataFusion query engine via the
    ``icefalldb query`` CLI, or ``engine="icefalldb"`` to route SELECTs through
    DuckDB and mutations through the native engine.
    """
    return attach(
        db_path,
        connection=connection,
        tables=[table],
        verify_data_checksums=verify_data_checksums,
        engine=engine,
    )


def _tokenize_sql(sql: str) -> list[tuple[str, int, int]]:
    """A minimal SQL tokenizer that ignores semicolons inside strings/comments."""
    tokens: list[tuple[str, int, int]] = []
    i = 0
    n = len(sql)
    while i < n:
        c = sql[i]
        if c.isspace():
            i += 1
            continue
        if c == "-" and i + 1 < n and sql[i + 1] == "-":
            start = i
            i += 2
            while i < n and sql[i] != "\n":
                i += 1
            tokens.append(("comment", start, i))
            continue
        if c == "/" and i + 1 < n and sql[i + 1] == "*":
            start = i
            i += 2
            while i + 1 < n and not (sql[i] == "*" and sql[i + 1] == "/"):
                i += 1
            i += 2
            tokens.append(("comment", start, i))
            continue
        if c in ("'", '"'):
            quote = c
            start = i
            i += 1
            while i < n:
                if sql[i] == quote:
                    if i + 1 < n and sql[i + 1] == quote:
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            tokens.append(("string", start, i))
            continue
        if c == ";":
            tokens.append(("semicolon", i, i + 1))
            i += 1
            continue
        start = i
        i += 1
        while i < n and not sql[i].isspace() and sql[i] != ";":
            i += 1
        kind = "word" if sql[start].isalpha() or sql[start] == "_" else "other"
        tokens.append((kind, start, i))
    return tokens


def _split_sql_statements(sql: str) -> list[str]:
    tokens = _tokenize_sql(sql)
    statements: list[str] = []
    stmt_start: int | None = None
    for kind, start, _end in tokens:
        if kind == "semicolon":
            if stmt_start is not None:
                statements.append(sql[stmt_start:start])
                stmt_start = None
        elif stmt_start is None and kind != "comment":
            stmt_start = start
    if stmt_start is not None:
        statements.append(sql[stmt_start:])
    return statements


def _extract_view_query(query: str) -> str:
    statements = _split_sql_statements(query)
    non_empty = [s.strip() for s in statements if s.strip()]
    if not non_empty:
        raise IcefallDBError("view query must not be empty")
    if len(non_empty) > 1:
        raise IcefallDBError("view query must contain a single SELECT statement")
    first = non_empty[0]
    lower = first.lower()
    if not (lower.startswith("select") or lower.startswith("with")):
        raise IcefallDBError("view query must be a single SELECT statement")
    return first


# Statement-type classification for the hybrid engine router.
_MUTATION_KEYWORDS = frozenset({"DELETE", "UPDATE", "MERGE"})
_READ_KEYWORDS = frozenset(
    {
        "SELECT",
        "WITH",
        "VALUES",
        "TABLE",
        "SHOW",
        "DESCRIBE",
        "DESC",
        "EXPLAIN",
        "PRAGMA",
    }
)


def _first_sql_keyword(sql: str) -> str:
    """Return the uppercased leading keyword of the first statement in ``sql``.

    Reuses the adapter tokenizer so leading comments and whitespace are skipped.
    Raises if ``sql`` is empty or contains more than one statement (the hybrid
    router routes one statement per call to keep dispatch unambiguous).
    """
    statements = _split_sql_statements(sql)
    non_empty = [s for s in statements if s.strip()]
    if not non_empty:
        raise IcefallDBError("SQL must not be empty")
    if len(non_empty) > 1:
        raise IcefallDBError(
            "the hybrid engine accepts a single statement per call; split "
            "multiple statements and execute them individually"
        )
    first = non_empty[0].strip()
    token = first.split(None, 1)[0] if first else ""
    # Tolerate a leading "(" for parenthesised expressions, e.g. "(SELECT ...)".
    return token.upper().lstrip("(").rstrip(",")


def _classify_sql(sql: str) -> str:
    """Classify ``sql`` as ``select``/``delete``/``update``/``merge``/``insert``/``other``."""
    keyword = _first_sql_keyword(sql)
    if keyword in _READ_KEYWORDS:
        return "select"
    if keyword in _MUTATION_KEYWORDS:
        return keyword.lower()
    if keyword == "INSERT":
        return "insert"
    return "other"


def _extract_mutation_table(sql: str) -> Optional[str]:
    """Best-effort extraction of the target table from a DELETE/UPDATE/MERGE.

    Handles simple unquoted identifiers (``DELETE FROM t``, ``UPDATE t``,
    ``MERGE INTO t`` / ``MERGE t``). Returns None if it cannot be determined.
    """
    statements = _split_sql_statements(sql)
    non_empty = [s for s in statements if s.strip()]
    if not non_empty:
        return None
    first = non_empty[0].strip()
    upper = first.upper()
    if upper.startswith("DELETE"):
        m = re.match(r"DELETE\s+FROM\s+([A-Za-z_][A-Za-z0-9_]*)", first, re.IGNORECASE)
    elif upper.startswith("UPDATE"):
        m = re.match(r"UPDATE\s+([A-Za-z_][A-Za-z0-9_]*)", first, re.IGNORECASE)
    elif upper.startswith("MERGE"):
        m = re.match(
            r"MERGE\s+(?:INTO\s+)?([A-Za-z_][A-Za-z0-9_]*)", first, re.IGNORECASE
        )
    else:
        return None
    return m.group(1) if m else None


def _table_referenced_in_sql(sql: str, table: str) -> bool:
    """Return True if ``table`` appears as a whole-word identifier in ``sql``.

    Over-detection is safe (it steers the query to the native engine); under-
    detection is not, so the check is deliberately broad and case-insensitive.
    """
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(table) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, sql, re.IGNORECASE) is not None


# Aggregate functions the native engine can serve from sidecar metadata.
# COUNT/MIN/MAX/COUNT(col) always work (ColumnStats is always present); the
# SUM/AVG/VAR/STDDEV family additionally needs a per-fragment .agg sidecar but,
# when absent, the native rule falls back to an unfiltered scan that is
# competitive with DuckDB, so including them is safe.
_METADATA_AGG_FUNCS = frozenset(
    {
        "count",
        "min",
        "max",
        "sum",
        "avg",
        "mean",
        "var",
        "var_pop",
        "var_samp",
        "stddev",
        "stddev_pop",
        "stddev_samp",
        "approx_distinct",
        "approx_percentile_cont",
    }
)

# Matches a bare ``SELECT <select-list> FROM <table>`` with nothing after the
# table reference (so WHERE / GROUP BY / JOIN / HAVING / LIMIT / ... all fail to
# match and fall through to DuckDB).
_METADATA_AGG_RE = re.compile(
    r"^\s*SELECT\s+(?P<select>.+?)\s+FROM\s+(?P<table>[A-Za-z_][\w.]*)\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)

# Matches a single select-list item that is an aggregate call with an optional
# alias, e.g. ``COUNT(*)``, ``min(value)``, ``AVG(amount) AS a``. Arguments are
# restricted to ``*``, bare identifiers, or numeric literals so that expressions
# (``SUM(value * 2)``, ``SUM(CASE WHEN ...)``) are NOT misclassified as
# metadata-eligible — those need a real scan and must stay on DuckDB.
_AGG_ITEM_RE = re.compile(
    r"^\s*(?P<func>\w+)\s*\(\s*(?P<args>\*|(?:[\w.]+|[-0-9.]+)(?:\s*,\s*(?:[\w.]+|[-0-9.]+))*)\s*\)"
    r"\s*(?:AS\s+)?(?P<alias>\"[^\"]+\"|\w+)?\s*$",
    re.IGNORECASE,
)


def _split_top_level_commas(text: str) -> list[str]:
    """Split ``text`` on commas that are not nested inside parentheses."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def _metadata_aggregate_target(sql: str) -> Optional[str]:
    """Return the table name if ``sql`` is an unfiltered single-table aggregate.

    Detects ``SELECT <agg, ...> FROM <table>`` with no WHERE/GROUP BY/JOIN/
    HAVING/LIMIT or other clause, where every select-list item is an aggregate
    function call. Returns the table name, or None if the query is not a
    metadata-served aggregate shape. Detection is deliberately conservative:
    a miss merely leaves the query on DuckDB (correct, just not accelerated).
    """
    statements = _split_sql_statements(sql)
    non_empty = [s for s in statements if s.strip()]
    if len(non_empty) != 1:
        return None
    m = _METADATA_AGG_RE.match(non_empty[0])
    if not m:
        return None
    select = m.group("select")
    items = _split_top_level_commas(select)
    if not items:
        return None
    for item in items:
        im = _AGG_ITEM_RE.match(item)
        if not im:
            return None
        if im.group("func").lower() not in _METADATA_AGG_FUNCS:
            return None
        if "distinct" in im.group("args").lower():
            return None  # DISTINCT aggregates are not metadata-served
    return m.group("table").split(".")[-1]


def create_view(db_path: str | Path, view_name: str, query: str) -> Path:
    """Create a materialized view definition under `views/<view_name>.sql`.

    The view query is stored as plain SQL and is materialized on demand by
    calling [`refresh_view`]. The view name must be a safe identifier.
    """
    db_path = Path(db_path).resolve()
    _validate_table_name(view_name)

    normalized = _extract_view_query(query)

    views_dir = db_path / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    view_file = views_dir / f"{view_name}.sql"
    view_file.write_text(normalized, encoding="utf-8")
    return view_file


def _find_icefalldb_cli() -> Optional[str]:
    """Locate the `icefalldb` CLI executable.

    Honors the `ICEFALLDB_CLI` environment variable, then searches `PATH`, and
    finally checks common Cargo build locations relative to the current working
    directory (`target/debug/icefalldb` and `target/release/icefalldb`).
    """
    env_cli = os.environ.get("ICEFALLDB_CLI")
    if env_cli:
        return env_cli
    path_cli = shutil.which("icefalldb")
    if path_cli:
        return path_cli
    cwd = Path.cwd()
    # Prefer the release build so benchmarks and production callers do not
    # accidentally use a debug binary.
    for candidate in ("target/release/icefalldb", "target/debug/icefalldb"):
        full = cwd / candidate
        if full.is_file() and os.access(str(full), os.X_OK):
            return str(full)
    return None


def _cli_env() -> dict[str, str]:
    """Return an environment dict for CLI subprocess calls.

    If the current Python interpreter is inside a virtual environment that also
    contains a `duckdb` executable (the DuckDB CLI), prepend that bin directory
    to PATH. This lets the Rust CLI find DuckDB when the venv is used without
    being explicitly activated.
    """
    env = os.environ.copy()
    venv_bin = Path(sys.executable).parent
    if (venv_bin / "duckdb").exists():
        existing = env.get("PATH", "")
        env["PATH"] = f"{venv_bin}{os.pathsep}{existing}" if existing else str(venv_bin)
    return env


def refresh_view(
    db_path: str | Path,
    view_name: str,
    connection: Optional[duckdb.DuckDBPyConnection] = None,
) -> None:
    """Refresh a materialized view by delegating to the `icefalldb refresh-view` CLI.

    The Rust CLI acquires the writer lock on the derived table for the entire
    operation, reads source data from committed manifests only, and atomically
    replaces the view table's latest snapshot with the query result. The
    `connection` argument is accepted for API compatibility but is not used for
    the refresh itself, because the CLI manages its own DuckDB session.
    """
    del connection  # unused: the CLI owns the refresh transaction end-to-end

    db_path = Path(db_path).resolve()
    _validate_table_name(view_name)

    view_file = db_path / "views" / f"{view_name}.sql"
    if not view_file.is_file():
        raise IcefallDBError(f"view '{view_name}' not found at {view_file}")
    query = view_file.read_text(encoding="utf-8").strip()
    if not query:
        raise IcefallDBError(f"view '{view_name}' query is empty")

    cli = _find_icefalldb_cli()
    if cli is None:
        raise IcefallDBError(
            "icefalldb CLI not found; set the ICEFALLDB_CLI environment variable or "
            "ensure icefalldb is on PATH"
        )

    result = subprocess.run(
        [cli, "refresh-view", str(db_path), view_name],
        capture_output=True,
        text=True,
        check=False,
        env=_cli_env(),
    )
    if result.returncode != 0:
        raise IcefallDBError(
            f"icefalldb refresh-view failed for view '{view_name}': {result.stderr}"
        )


def import_tsv(
    db_path: str | Path,
    table: str,
    file: str | Path,
    *,
    encrypt: bool = False,
    encrypt_columns: Optional[Iterable[str]] = None,
    encrypt_footer: bool = False,
    key_file: Optional[str | Path] = None,
) -> None:
    """Import a TSV file into a IcefallDB table using the `icefalldb import` CLI.

    If `table` does not exist, the CLI infers a schema from the TSV header and
    sample rows and creates the table before importing.

    To create an **encrypted** table, set ``encrypt=True`` (encrypt the whole
    table with the footer key) and/or pass ``encrypt_columns=["ssn", ...]`` (a
    per-column key for each named column, the rest left plaintext). Keys are
    read from ``key_file`` (a JSON ``{"keys": {...}}`` file) or, by default,
    from ``ICEFALLDB_KEY_*`` environment variables. Read the table back with
    ``attach(db, key_file=...)`` or the same env vars.
    """
    db_path = Path(db_path).resolve()
    file = Path(file).resolve()
    _validate_table_name(table)

    cli = _find_icefalldb_cli()
    if cli is None:
        raise IcefallDBError(
            "icefalldb CLI not found; set the ICEFALLDB_CLI environment variable or "
            "ensure icefalldb is on PATH"
        )

    cmd = [cli, "import", str(db_path), table, str(file)]
    if encrypt:
        cmd.append("--encrypt")
    for col in encrypt_columns or []:
        cmd.extend(["--encrypt-column", col])
    if encrypt_footer:
        cmd.append("--encrypt-footer")
    if key_file is not None:
        cmd.extend(["--key-file", str(key_file)])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=_cli_env(),
    )
    if result.returncode != 0:
        raise IcefallDBError(
            f"icefalldb import failed for table '{table}': {result.stderr}"
        )


def export_tsv(
    db_path: str | Path,
    table: str,
    file: str | Path,
) -> None:
    """Export a IcefallDB table to a TSV file using the `icefalldb export` CLI."""
    db_path = Path(db_path).resolve()
    file = Path(file).resolve()
    _validate_table_name(table)

    cli = _find_icefalldb_cli()
    if cli is None:
        raise IcefallDBError(
            "icefalldb CLI not found; set the ICEFALLDB_CLI environment variable or "
            "ensure icefalldb is on PATH"
        )

    result = subprocess.run(
        [cli, "export", str(db_path), table, str(file)],
        capture_output=True,
        text=True,
        check=False,
        env=_cli_env(),
    )
    if result.returncode != 0:
        raise IcefallDBError(
            f"icefalldb export failed for table '{table}': {result.stderr}"
        )


def _to_arrow_table(table: object) -> pa.Table:
    """Normalize `table` to a PyArrow Table."""
    if isinstance(table, pa.Table):
        return table
    if hasattr(table, "to_arrow_table"):
        return table.to_arrow_table()
    if hasattr(table, "to_arrow"):
        result = table.to_arrow()
        if isinstance(result, pa.Table):
            return result
    con = duckdb.connect()
    try:
        con.register("input_table", table)
        return con.sql("SELECT * FROM input_table").to_arrow_table()
    finally:
        con.close()


def _is_complex_arrow_type(t: pa.DataType) -> bool:
    return (
        pa.types.is_list(t)
        or pa.types.is_large_list(t)
        or pa.types.is_struct(t)
        or pa.types.is_map(t)
        or pa.types.is_decimal128(t)
        or pa.types.is_fixed_size_binary(t)
    )


def _arrow_value_to_json(value: object, arrow_type: pa.DataType) -> object:
    """Recursively convert a PyArrow value to a JSON-serializable object.

    Uses the Arrow type to handle maps (list-of-tuples/dicts), lists, structs,
    and scalar conversions (bytes, timestamps, decimals) at any nesting level.
    """
    if value is None:
        return None
    if pa.types.is_map(arrow_type):
        # PyArrow represents a map value as a list of (key, value) tuples or,
        # in some contexts, as a list of {"key": ..., "value": ...} dicts.
        result: dict[str, object] = {}
        for entry in value:
            if isinstance(entry, dict):
                key = entry["key"]
                item = entry["value"]
            else:
                key, item = entry
            result[str(key)] = _arrow_value_to_json(item, arrow_type.item_type)
        return result
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return [_arrow_value_to_json(v, arrow_type.value_type) for v in value]
    if pa.types.is_struct(arrow_type):
        return {
            field.name: _arrow_value_to_json(value[field.name], field.type)
            for field in arrow_type
        }
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return str(value)
    return value


def _format_cell(value: object, arrow_type: pa.DataType) -> str:
    """Format a single PyArrow value as a IcefallDB TSV cell string."""
    if _is_complex_arrow_type(arrow_type):
        return json.dumps(
            _arrow_value_to_json(value, arrow_type),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    if pa.types.is_boolean(arrow_type):
        return "true" if value else "false"
    if pa.types.is_integer(arrow_type):
        return str(value)
    if pa.types.is_floating(arrow_type):
        return repr(value)
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return str(value)
    if (
        pa.types.is_binary(arrow_type)
        or pa.types.is_large_binary(arrow_type)
        or pa.types.is_fixed_size_binary(arrow_type)
    ):
        if isinstance(value, bytes):
            return base64.b64encode(value).decode("ascii")
        return str(value)
    if pa.types.is_temporal(arrow_type):
        if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
            return value.isoformat()
        return str(value)
    return str(value)


def _escape_tsv(value: str) -> str:
    """Escape tabs, newlines, carriage returns, and backslashes."""
    return (
        value.replace("\\", "\\\\")
        .replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def _unescape_tsv(value: str) -> str:
    """Unescape IcefallDB TSV escape sequences."""
    out = []
    i = 0
    while i < len(value):
        if value[i] == "\\" and i + 1 < len(value):
            nxt = value[i + 1]
            if nxt == "t":
                out.append("\t")
            elif nxt == "n":
                out.append("\n")
            elif nxt == "r":
                out.append("\r")
            elif nxt == "\\":
                out.append("\\")
            else:
                raise ValueError(f"unknown escape sequence \\{nxt}")
            i += 2
        else:
            out.append(value[i])
            i += 1
    return "".join(out)


def _split_tsv_line(line: str) -> list[str]:
    """Split a TSV line on tabs while respecting IcefallDB backslash escapes."""
    cells = []
    current = []
    i = 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line):
            current.append(line[i])
            current.append(line[i + 1])
            i += 2
        elif line[i] == "\t":
            cells.append("".join(current))
            current = []
            i += 1
        else:
            current.append(line[i])
            i += 1
    cells.append("".join(current))
    return cells


def _infer_column_type(values: list[str]) -> tuple[pa.DataType, list[object]]:
    """Infer a PyArrow type from a column of unescaped string values."""
    non_null = [v for v in values if v != ""]
    if not non_null:
        return pa.string(), [None if v == "" else v for v in values]

    # Integer.
    try:
        parsed: list[object | None] = []
        for v in values:
            if v == "":
                parsed.append(None)
            else:
                parsed.append(int(v))
        return pa.int64(), parsed
    except ValueError:
        pass

    # Float.
    try:
        parsed = []
        for v in values:
            if v == "":
                parsed.append(None)
            else:
                parsed.append(float(v))
        return pa.float64(), parsed
    except ValueError:
        pass

    # Boolean (strict: only "true" and "false").
    bool_map = {"true": True, "false": False}
    try:
        parsed = []
        for v in values:
            if v == "":
                parsed.append(None)
            else:
                if v not in bool_map:
                    raise ValueError
                parsed.append(bool_map[v])
        return pa.bool_(), parsed
    except ValueError:
        pass

    # Default to string.
    return pa.string(), [None if v == "" else v for v in values]


def read_tsv(path: str | Path) -> pa.Table:
    """Read a IcefallDB-format TSV file into a PyArrow table.

    This is a general-purpose helper that does not require a IcefallDB context.
    Column types are inferred from the unescaped cell values. Because inference
    is schema-less, complex-typed JSON cells (``list<T>``, ``struct<...>``,
    ``map<K,V>``, etc.) are returned as JSON strings rather than parsed Arrow
    values.
    """
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8", newline="") as f:
        lines = f.read().splitlines()
    if not lines:
        return pa.table({})

    header = [_unescape_tsv(cell) for cell in _split_tsv_line(lines[0])]
    data_rows: list[list[str]] = []
    for line_no, line in enumerate(lines[1:], start=2):
        cells = _split_tsv_line(line)
        if len(cells) != len(header):
            raise ValueError(
                f"TSV line {line_no}: expected {len(header)} fields, found {len(cells)}"
            )
        data_rows.append(cells)

    if not data_rows:
        return pa.table({name: pa.array([], type=pa.string()) for name in header})

    arrays = {}
    for i, name in enumerate(header):
        unescaped = [_unescape_tsv(row[i]) for row in data_rows]
        arrow_type, parsed = _infer_column_type(unescaped)
        arrays[name] = pa.array(parsed, type=arrow_type)

    return pa.table(arrays)


def write_tsv(table: object, path: str | Path) -> None:
    """Write a PyArrow table or DuckDB relation to a IcefallDB-format TSV file.

    This is a general-purpose helper that does not require a IcefallDB context.
    Tabs, newlines, carriage returns, and backslashes are escaped, and NULL
    values are written as empty fields. Complex-typed columns (``list<T>``,
    ``struct<...>``, ``map<K,V>``, etc.) are serialized as JSON-in-TSV so they
    can be imported by IcefallDB's schema-aware TSV importer. The companion
    [`read_tsv`] is schema-less and will infer those cells as JSON strings.
    """
    path = Path(path).resolve()
    arrow_table = _to_arrow_table(table)
    schema = arrow_table.schema

    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("\t".join(_escape_tsv(field.name) for field in schema) + "\n")
        for batch in arrow_table.to_batches():
            columns = [batch.column(i) for i in range(batch.num_columns)]
            types = [schema.field(i).type for i in range(batch.num_columns)]
            for row_idx in range(batch.num_rows):
                cells = []
                for col_idx, col in enumerate(columns):
                    if col.is_null()[row_idx]:
                        cells.append("")
                    else:
                        value = col[row_idx].as_py()
                        cell = _format_cell(value, types[col_idx])
                        cells.append(_escape_tsv(cell))
                f.write("\t".join(cells) + "\n")
