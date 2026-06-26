use crate::{IcefallDBError, Result};
use std::path::{Component, Path};

/// Validate a storage path.
///
/// Rejects empty paths, absolute paths, and any `..` component. Returns a
/// cleaned relative path using `/` as the separator.
pub fn validate_path(raw: &str) -> Result<String> {
    if raw.is_empty() {
        return Err(IcefallDBError::InvalidPath(
            "path cannot be empty".to_string(),
        ));
    }

    let path = Path::new(raw);
    if path.is_absolute() {
        return Err(IcefallDBError::InvalidPath(format!(
            "absolute paths are not allowed: {raw}"
        )));
    }

    let mut cleaned = String::new();
    for component in path.components() {
        match component {
            Component::Normal(component) => {
                let Some(component) = component.to_str() else {
                    return Err(IcefallDBError::InvalidPath(format!(
                        "path contains invalid characters: {raw}"
                    )));
                };
                if !cleaned.is_empty() {
                    cleaned.push('/');
                }
                cleaned.push_str(component);
            }
            Component::CurDir => continue,
            Component::ParentDir => {
                return Err(IcefallDBError::InvalidPath(format!(
                    "path traversal is not allowed: {raw}"
                )));
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(IcefallDBError::InvalidPath(format!(
                    "absolute paths are not allowed: {raw}"
                )));
            }
        }
    }

    if cleaned.is_empty() {
        return Err(IcefallDBError::InvalidPath(format!(
            "path cannot be empty: {raw}"
        )));
    }

    Ok(cleaned)
}
