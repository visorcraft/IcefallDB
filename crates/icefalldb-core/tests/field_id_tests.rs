use icefalldb_core::metadata::{Column, Schema};

fn make_schema(columns: Vec<Column>) -> Schema {
    Schema {
        schema_id: 1,
        columns,
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 1000,
        row_group_target_bytes: 1024 * 1024,
        dropped_columns: vec![],
        max_field_id: 0,
    }
}

#[test]
fn test_assign_field_ids_initially() {
    let mut schema = make_schema(vec![
        Column::new("a", "int64", false),
        Column::new("b", "utf8", true),
        Column::new("c", "bool", true),
    ]);
    schema.assign_field_ids(None);

    assert_eq!(schema.columns[0].field_id, 1);
    assert_eq!(schema.columns[1].field_id, 2);
    assert_eq!(schema.columns[2].field_id, 3);
    assert_eq!(schema.max_field_id, 3);
    assert!(schema.has_field_ids());
}

#[test]
fn test_assign_field_ids_preserves_existing_ids() {
    let mut previous = make_schema(vec![
        Column {
            name: "a".into(),
            r#type: "int64".into(),
            nullable: false,
            field_id: 1,
        },
        Column {
            name: "b".into(),
            r#type: "utf8".into(),
            nullable: true,
            field_id: 2,
        },
    ]);
    previous.max_field_id = 2;

    let mut current = make_schema(vec![
        Column::new("a", "int64", false),
        Column::new("b", "utf8", true),
        Column::new("c", "bool", true),
    ]);
    current.assign_field_ids(Some(&previous));

    assert_eq!(current.columns[0].field_id, 1);
    assert_eq!(current.columns[1].field_id, 2);
    assert_eq!(current.columns[2].field_id, 3);
    assert_eq!(current.max_field_id, 3);
}

#[test]
fn test_assign_field_ids_never_reuses_dropped_ids() {
    let mut previous = make_schema(vec![
        Column {
            name: "a".into(),
            r#type: "int64".into(),
            nullable: false,
            field_id: 1,
        },
        Column {
            name: "b".into(),
            r#type: "utf8".into(),
            nullable: true,
            field_id: 2,
        },
    ]);
    previous.dropped_columns.push("b".into());
    previous.max_field_id = 2;

    let mut current = make_schema(vec![
        Column::new("a", "int64", false),
        Column::new("c", "bool", true),
    ]);
    current.assign_field_ids(Some(&previous));

    assert_eq!(current.columns[0].field_id, 1);
    assert_eq!(current.columns[1].field_id, 3);
    assert_eq!(current.max_field_id, 3);
}

#[test]
fn test_repair_field_ids() {
    let mut schema = make_schema(vec![
        Column::new("x", "int64", false),
        Column::new("y", "utf8", true),
    ]);
    schema.repair_field_ids();

    assert_eq!(schema.columns[0].field_id, 1);
    assert_eq!(schema.columns[1].field_id, 2);
    assert_eq!(schema.max_field_id, 2);
}

#[test]
fn test_has_field_ids_false_for_unassigned() {
    let schema = make_schema(vec![Column::new("x", "int64", false)]);
    assert!(!schema.has_field_ids());
}

#[test]
fn test_next_field_id_accounts_for_dropped_columns() {
    let mut schema = make_schema(vec![Column {
        name: "a".into(),
        r#type: "int64".into(),
        nullable: false,
        field_id: 1,
    }]);
    schema.max_field_id = 5;

    assert_eq!(schema.next_field_id(), 6);
}

#[test]
fn test_next_field_id_accounts_for_existing_columns() {
    let schema = make_schema(vec![Column {
        name: "a".into(),
        r#type: "int64".into(),
        nullable: false,
        field_id: 7,
    }]);

    assert_eq!(schema.next_field_id(), 8);
}
