//! Production metadata required to bind annotated lookup columns.
//!
//! Circuit-debugging helpers (failure reporting, cost and tracing support)
//! are intentionally not part of this formal vendor.  The
//! column key remains in the production `plonk` module because annotations
//! are consumed while configuring and serializing a real circuit.

use super::Any;

/// Stable key for a concrete circuit column annotation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Column {
    /// The type of the column.
    pub(super) column_type: Any,
    /// The index of the column.
    pub(super) index: usize,
}

impl Column {
    /// Return the column type.
    pub fn column_type(&self) -> Any {
        self.column_type
    }

    /// Return the column index.
    pub fn index(&self) -> usize {
        self.index
    }
}

impl From<(Any, usize)> for Column {
    fn from((column_type, index): (Any, usize)) -> Self {
        Self { column_type, index }
    }
}

impl From<super::Column<Any>> for Column {
    fn from(column: super::Column<Any>) -> Self {
        Self {
            column_type: *column.column_type(),
            index: column.index(),
        }
    }
}
