use arrow::array::{
    Array, BooleanArray, Decimal128Array, Float64Array, Int64Array, ListArray, MapArray,
    RecordBatch, StringArray, StructArray,
};
use arrow::datatypes::{DataType, Field, Fields};
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::tsv::{decode_tsv, encode_tsv};
use std::sync::Arc;

fn make_flat_schema() -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![
            Column {
                name: "id".into(),
                r#type: "int64".into(),
                nullable: false,
                field_id: 0,
            },
            Column {
                name: "name".into(),
                r#type: "utf8".into(),
                nullable: true,
                field_id: 0,
            },
            Column {
                name: "active".into(),
                r#type: "bool".into(),
                nullable: true,
                field_id: 0,
            },
            Column {
                name: "score".into(),
                r#type: "float64".into(),
                nullable: true,
                field_id: 0,
            },
            Column {
                name: "ts".into(),
                r#type: "timestamp[us]".into(),
                nullable: true,
                field_id: 0,
            },
        ],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 1000,
        row_group_target_bytes: 1024 * 1024,
        dropped_columns: vec![],
        max_field_id: 0,
    };
    schema.assign_field_ids(None);
    schema
}

fn make_flat_batch() -> RecordBatch {
    let arrow_schema = make_flat_schema().arrow_schema().unwrap();
    let id = Int64Array::from(vec![1, 2, 3]);
    let name = StringArray::from(vec![Some("alice"), None, Some("bob\ttab")]);
    let active = BooleanArray::from(vec![Some(true), Some(false), None]);
    let score = Float64Array::from(vec![Some(1.5), None, Some(3.0)]);
    let ts = arrow::array::TimestampMicrosecondArray::from(vec![
        Some(1704067200000000),
        None,
        Some(1704153600000000),
    ]);
    RecordBatch::try_new(
        Arc::new(arrow_schema),
        vec![
            Arc::new(id),
            Arc::new(name),
            Arc::new(active),
            Arc::new(score),
            Arc::new(ts),
        ],
    )
    .unwrap()
}

#[test]
fn test_roundtrip_flat_table() {
    let schema = make_flat_schema();
    let batch = make_flat_batch();
    let bytes = encode_tsv(&batch).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();

    let expected_header = "id\tname\tactive\tscore\tts";
    assert!(
        text.starts_with(expected_header),
        "header mismatch: {}",
        text
    );
    assert!(text.contains("bob\\ttab"));

    let decoded = decode_tsv(&bytes, &schema).unwrap();
    assert_eq!(decoded.len(), 1);
    let decoded = &decoded[0];
    assert_eq!(decoded.num_rows(), 3);
    assert_eq!(decoded.num_columns(), 5);

    let ids = decoded
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(ids.values(), &[1, 2, 3]);

    let names = decoded
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names.value(0), "alice");
    assert!(names.is_null(1));
    assert_eq!(names.value(2), "bob\ttab");
}

#[test]
fn test_null_and_escaping_roundtrip() {
    let schema = Schema {
        schema_id: 1,
        columns: vec![
            Column {
                name: "id".into(),
                r#type: "int64".into(),
                nullable: false,
                field_id: 0,
            },
            Column {
                name: "text".into(),
                r#type: "utf8".into(),
                nullable: true,
                field_id: 0,
            },
        ],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 100,
        row_group_target_bytes: 1024 * 1024,
        dropped_columns: vec![],
        max_field_id: 0,
    };

    let arrow_schema = schema.arrow_schema().unwrap();
    let id = Int64Array::from(vec![1, 2, 3, 4, 5]);
    let text = StringArray::from(vec![
        Some(""),
        Some("a\tb"),
        Some("a\nb"),
        Some("a\\b"),
        None,
    ]);
    let batch =
        RecordBatch::try_new(Arc::new(arrow_schema), vec![Arc::new(id), Arc::new(text)]).unwrap();

    let bytes = encode_tsv(&batch).unwrap();
    let decoded = decode_tsv(&bytes, &schema).unwrap();
    let decoded = &decoded[0];
    let out = decoded
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(out.value(0), "");
    assert_eq!(out.value(1), "a\tb");
    assert_eq!(out.value(2), "a\nb");
    assert_eq!(out.value(3), "a\\b");
    assert!(out.is_null(4));
}

#[test]
fn test_roundtrip_list_struct_map() {
    let schema = Schema {
        schema_id: 1,
        columns: vec![
            Column {
                name: "tags".into(),
                r#type: "list<utf8>".into(),
                nullable: true,
                field_id: 0,
            },
            Column {
                name: "point".into(),
                r#type: "struct<x:int64,y:int64>".into(),
                nullable: true,
                field_id: 0,
            },
            Column {
                name: "attrs".into(),
                r#type: "map<utf8,int64>".into(),
                nullable: true,
                field_id: 0,
            },
            Column {
                name: "price".into(),
                r#type: "decimal128(10,2)".into(),
                nullable: true,
                field_id: 0,
            },
        ],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 100,
        row_group_target_bytes: 1024 * 1024,
        dropped_columns: vec![],
        max_field_id: 0,
    };

    let arrow_schema = schema.arrow_schema().unwrap();

    let tags_values = StringArray::from(vec!["a", "b", "c"]);
    let tags_offsets =
        arrow::buffer::OffsetBuffer::new(arrow::buffer::ScalarBuffer::from(vec![0i32, 2, 2, 3]));
    let tags = ListArray::new(
        Arc::new(Field::new("item", DataType::Utf8, true)),
        tags_offsets,
        Arc::new(tags_values),
        None,
    );

    let point_fields = Fields::from(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("y", DataType::Int64, true),
    ]);
    let x = Int64Array::from(vec![Some(1), None, Some(3)]);
    let y = Int64Array::from(vec![Some(2), Some(3), None]);
    let point = StructArray::new(point_fields, vec![Arc::new(x), Arc::new(y)], None);

    let entry_fields = Fields::from(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("value", DataType::Int64, true),
    ]);
    let key = StringArray::from(vec!["k1", "k2"]);
    let value = Int64Array::from(vec![Some(10), Some(5)]);
    let entries = StructArray::new(entry_fields, vec![Arc::new(key), Arc::new(value)], None);
    let offsets =
        arrow::buffer::OffsetBuffer::new(arrow::buffer::ScalarBuffer::from(vec![0i32, 1, 1, 2]));
    let attrs = MapArray::new(
        Field::new("entries", DataType::Struct(entries.fields().clone()), false).into(),
        offsets,
        entries,
        None,
        false,
    );

    let price = Decimal128Array::from(vec![Some(12345), None, Some(-678)])
        .with_data_type(DataType::Decimal128(10, 2));

    let batch = RecordBatch::try_new(
        Arc::new(arrow_schema),
        vec![
            Arc::new(tags),
            Arc::new(point),
            Arc::new(attrs),
            Arc::new(price),
        ],
    )
    .unwrap();

    let bytes = encode_tsv(&batch).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert!(text.contains("tags\tpoint\tattrs\tprice"));
    assert!(text.contains(r#"["a","b"]"#));
    assert!(text.contains(r#"{"x":1,"y":2}"#));
    assert!(text.contains(r#"{"k1":10}"#));
    assert!(text.contains(r#"123.45"#));

    let decoded = decode_tsv(&bytes, &schema).unwrap();
    assert_eq!(decoded.len(), 1);
    let decoded = &decoded[0];
    assert_eq!(decoded.num_rows(), 3);
    assert_eq!(decoded.num_columns(), 4);
}

#[test]
fn test_header_missing_column_error() {
    let schema = Schema {
        schema_id: 1,
        columns: vec![
            Column {
                name: "id".into(),
                r#type: "int64".into(),
                nullable: false,
                field_id: 0,
            },
            Column {
                name: "name".into(),
                r#type: "utf8".into(),
                nullable: true,
                field_id: 0,
            },
        ],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 100,
        row_group_target_bytes: 1024 * 1024,
        dropped_columns: vec![],
        max_field_id: 0,
    };

    let tsv = "id\n1\n";
    let err = decode_tsv(tsv.as_bytes(), &schema).unwrap_err();
    let msg = format!("{}", err);
    assert!(
        msg.contains("name"),
        "error should mention missing column: {}",
        msg
    );
    assert!(
        msg.contains("missing"),
        "error should mention missing: {}",
        msg
    );
}

#[test]
fn test_invalid_primitive_value_error() {
    let schema = Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "id".into(),
            r#type: "int64".into(),
            nullable: false,
            field_id: 0,
        }],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 100,
        row_group_target_bytes: 1024 * 1024,
        dropped_columns: vec![],
        max_field_id: 0,
    };

    let tsv = "id\nnot_a_number\n";
    let err = decode_tsv(tsv.as_bytes(), &schema).unwrap_err();
    let msg = format!("{}", err);
    assert!(
        msg.contains("invalid int64"),
        "error should mention invalid int64: {}",
        msg
    );
    assert!(
        msg.contains("not_a_number"),
        "error should include value: {}",
        msg
    );
}

#[test]
fn test_too_few_fields_error() {
    let schema = make_flat_schema();
    // Header has 5 columns; data row has 2.
    let tsv = "id\tname\tactive\tscore\tts\n1\t1.0\n";
    let err = decode_tsv(tsv.as_bytes(), &schema).unwrap_err();
    let msg = format!("{}", err);
    assert!(
        msg.contains("expected 5 fields, found 2"),
        "error should report expected/found counts: {}",
        msg
    );
    assert!(
        msg.contains("line 2"),
        "error should report line number: {}",
        msg
    );
}

#[test]
fn test_too_many_fields_error() {
    let schema = make_flat_schema();
    // Header has 5 columns; data row has 6.
    let tsv = "id\tname\tactive\tscore\tts\n1\t1.0\tone\textra\tm\t2024-01-01T00:00:00Z\n";
    let err = decode_tsv(tsv.as_bytes(), &schema).unwrap_err();
    let msg = format!("{}", err);
    assert!(
        msg.contains("expected 5 fields, found 6"),
        "error should report expected/found counts: {}",
        msg
    );
    assert!(
        msg.contains("line 2"),
        "error should report line number: {}",
        msg
    );
}
