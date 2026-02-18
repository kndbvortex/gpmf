//! Rust data structures for gradual pattern mining
//!
//! This module provides Rust equivalents of Python datatypes with PyO3 bindings.

use pyo3::prelude::*;

/// Direction of variation for a gradual item
#[pyclass(eq, eq_int)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Variation {
    /// Positive variation ("+")
    Increase,
    /// Negative variation ("-")
    Decrease,
}

#[pymethods]
impl Variation {
    fn __str__(&self) -> &str {
        match self {
            Variation::Increase => "+",
            Variation::Decrease => "-",
        }
    }

    fn __repr__(&self) -> String {
        format!("Variation.{:?}", self)
    }
}

/// A gradual item represents an attribute with a direction of variation
///
/// For example, "Age+" means increasing age, "Salary-" means decreasing salary.
/// Items are encoded as: attribute_index * 2 + (0 for +, 1 for -)
#[pyclass]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GradualItem {
    #[pyo3(get)]
    pub attribute_index: usize,
    #[pyo3(get)]
    pub variation: Variation,
}

#[pymethods]
impl GradualItem {
    #[new]
    fn new(attribute_index: usize, variation: Variation) -> Self {
        GradualItem {
            attribute_index,
            variation,
        }
    }

    /// Convert to integer encoding: 2*k for +, 2*k+1 for -
    fn to_code(&self) -> usize {
        self.attribute_index * 2
            + match self.variation {
                Variation::Increase => 0,
                Variation::Decrease => 1,
            }
    }

    /// Create from integer encoding
    #[staticmethod]
    fn from_code(code: usize) -> Self {
        GradualItem {
            attribute_index: code / 2,
            variation: if code % 2 == 0 {
                Variation::Increase
            } else {
                Variation::Decrease
            },
        }
    }

    fn __str__(&self) -> String {
        format!(
            "{}{}",
            self.attribute_index + 1,
            match self.variation {
                Variation::Increase => "+",
                Variation::Decrease => "-",
            }
        )
    }

    fn __repr__(&self) -> String {
        format!("GradualItem({}{})", self.attribute_index + 1, self.__str__())
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    fn __eq__(&self, other: &GradualItem) -> bool {
        self.attribute_index == other.attribute_index && self.variation == other.variation
    }
}

/// Result of gradual pattern mining (returned to Python)
#[pyclass]
#[derive(Clone, Debug)]
pub struct GradualPatternResult {
    #[pyo3(get)]
    pub items: Vec<GradualItem>,
    #[pyo3(get)]
    pub support: usize,
    #[pyo3(get)]
    pub item_codes: Vec<usize>,
}

#[pymethods]
impl GradualPatternResult {
    #[new]
    pub fn new(items: Vec<GradualItem>, support: usize) -> Self {
        let item_codes = items.iter().map(|i| i.to_code()).collect();
        GradualPatternResult {
            items,
            support,
            item_codes,
        }
    }

    pub fn __str__(&self) -> String {
        let item_strs: Vec<_> = self.items.iter().map(|i| i.__str__()).collect();
        format!("{{{}}}, sup={}", item_strs.join(", "), self.support)
    }

    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}

/// Represents a pair of transactions (tid1, tid2)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TransactionPair {
    pub tid1: usize,
    pub tid2: usize,
}

impl TransactionPair {
    pub fn new(tid1: usize, tid2: usize) -> Self {
        TransactionPair { tid1, tid2 }
    }
}

/// A virtual transaction created from a pair of original transactions
#[derive(Clone, Debug)]
pub struct VirtualTransaction {
    pub items: Vec<usize>,  // List of gradual item codes
    pub pair: TransactionPair,
    pub vtid: usize,        // Virtual transaction ID
}

impl VirtualTransaction {
    pub fn new(items: Vec<usize>, pair: TransactionPair, vtid: usize) -> Self {
        VirtualTransaction { items, pair, vtid }
    }
}

/// Helper to convert between Python and Rust variations
impl From<&str> for Variation {
    fn from(s: &str) -> Self {
        match s {
            "+" => Variation::Increase,
            "-" => Variation::Decrease,
            _ => panic!("Invalid variation: {}", s),
        }
    }
}

/// Convert Python list of codes to GradualItems
pub fn codes_to_items(codes: &[usize]) -> Vec<GradualItem> {
    codes.iter().map(|&code| GradualItem::from_code(code)).collect()
}

/// Convert GradualItems to codes
pub fn items_to_codes(items: &[GradualItem]) -> Vec<usize> {
    items.iter().map(|i| i.to_code()).collect()
}
