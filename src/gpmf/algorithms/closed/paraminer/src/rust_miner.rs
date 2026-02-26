//! Main Rust miner class for gradual pattern mining
//!
//! Provides the main API for Python integration.

use crate::rust_conversion::{build_item_matrices, convert_to_gradual_items};
use crate::rust_datatypes::{GradualPatternResult, TransactionPair};
use crate::rust_miner_engine::{mine_recursive, MiningContext};
use crate::BoolMatrix;
use pyo3::prelude::*;
use std::collections::HashSet;

/// Main Rust-based gradual pattern miner
#[pyclass]
pub struct RustGradualMiner {
    min_support: f64,
    num_threads: Option<usize>,
    verbose: bool,

    // Data (loaded)
    transactions: Vec<Vec<f64>>,
    num_transactions: usize,
    num_attributes: usize,
    threshold: usize,

    // Built during mining (stored to avoid recomputation)
    item_matrices: Vec<BoolMatrix>,
    item_to_vtids: Vec<Vec<usize>>,
    vtid_to_pair: Vec<TransactionPair>,
}

#[pymethods]
impl RustGradualMiner {
    /// Create a new Rust gradual miner
    ///
    /// # Arguments
    /// * `min_support` - Minimum support threshold (0-1 for relative, >1 for absolute)
    /// * `num_threads` - Number of threads for parallel processing (None = auto)
    /// * `verbose` - Print debug information during mining
    #[new]
    #[pyo3(signature = (min_support, num_threads=None, verbose=None))]
    fn new(min_support: f64, num_threads: Option<usize>, verbose: Option<bool>) -> Self {
        // Set number of threads if specified
        if let Some(threads) = num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .ok();
        }

        RustGradualMiner {
            min_support,
            num_threads,
            verbose: verbose.unwrap_or(false),
            transactions: Vec::new(),
            num_transactions: 0,
            num_attributes: 0,
            threshold: 0,
            item_matrices: Vec::new(),
            item_to_vtids: Vec::new(),
            vtid_to_pair: Vec::new(),
        }
    }

    /// Load transaction data
    ///
    /// # Arguments
    /// * `transactions` - List of transaction vectors (each is a list of floats)
    /// * `num_attributes` - Number of attributes per transaction
    fn load_transactions(&mut self, transactions: Vec<Vec<f64>>, num_attributes: usize) {
        self.transactions = transactions;
        self.num_transactions = self.transactions.len();
        self.num_attributes = num_attributes;

        // Compute threshold
        if self.min_support < 1.0 {
            // Relative threshold based on number of original transactions
            self.threshold = (self.min_support * self.num_transactions as f64).ceil() as usize;
            println!(
                "Relative min_support={} results in threshold={}",
                self.min_support, self.threshold
            );
        } else {
            // Absolute threshold
            self.threshold = self.min_support as usize;
        }

        if self.verbose {
            println!(
                "Loaded {} transactions with {} attributes",
                self.num_transactions, self.num_attributes
            );
            println!(
                "Mining with min_support={} (threshold={})",
                self.min_support, self.threshold
            );
        }
    }

    /// Mine closed frequent gradual patterns
    ///
    /// # Returns
    /// List of GradualPatternResult objects
    fn mine(&mut self) -> PyResult<Vec<GradualPatternResult>> {
        if self.transactions.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "No data loaded. Call load_transactions() first.",
            ));
        }

        // Step 1: Convert to gradual items
        if self.verbose {
            println!("Converting to gradual items...");
        }

        let (vtransactions, vtid_to_pair) =
            convert_to_gradual_items(&self.transactions, self.num_attributes);

        let num_vtrans = vtransactions.len();
        self.vtid_to_pair = vtid_to_pair;

        if self.verbose {
            println!("Created {} virtual transactions", num_vtrans);
        }

        // Step 2: Build boolean matrices
        if self.verbose {
            println!("Building boolean matrices...");
        }

        let num_items = self.num_attributes * 2;
        let (item_matrices, item_to_vtids) =
            build_item_matrices(&vtransactions, num_items, self.num_transactions);

        self.item_matrices = item_matrices;
        self.item_to_vtids = item_to_vtids;

        // Step 3: Start mining from empty pattern
        if self.verbose {
            println!("Mining patterns...");
        }

        let mut found_patterns: HashSet<Vec<usize>> = HashSet::new();
        let mut results: Vec<GradualPatternResult> = Vec::new();

        let empty_pattern = Vec::new();
        let all_vtids: HashSet<usize> = (0..num_vtrans).collect();
        let empty_exclusion = HashSet::new();

        let context = MiningContext {
            item_matrices: &self.item_matrices,
            item_to_vtids: &self.item_to_vtids,
            vtid_to_pair: &self.vtid_to_pair,
            num_attributes: self.num_attributes,
            threshold: self.threshold,
            verbose: self.verbose,
        };

        mine_recursive(
            empty_pattern,
            all_vtids,
            empty_exclusion,
            &context,
            &mut found_patterns,
            &mut results,
            0,
        );

        if self.verbose {
            println!("\nMining complete! Found {} patterns", results.len());
        }

        Ok(results)
    }

    /// Get the computed threshold
    fn get_threshold(&self) -> usize {
        self.threshold
    }

    /// Get the number of loaded transactions
    fn get_num_transactions(&self) -> usize {
        self.num_transactions
    }

    /// Get the number of attributes
    fn get_num_attributes(&self) -> usize {
        self.num_attributes
    }

    fn __repr__(&self) -> String {
        format!(
            "RustGradualMiner(min_support={}, num_transactions={}, num_attributes={})",
            self.min_support, self.num_transactions, self.num_attributes
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_miner_creation() {
        let miner = RustGradualMiner::new(0.3, None, Some(false));
        assert_eq!(miner.min_support, 0.3);
        assert_eq!(miner.num_transactions, 0);
    }

    #[test]
    fn test_load_transactions() {
        let mut miner = RustGradualMiner::new(0.5, None, Some(false));
        let transactions = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
        ];

        miner.load_transactions(transactions, 3);

        assert_eq!(miner.num_transactions, 3);
        assert_eq!(miner.num_attributes, 3);
        // 0.5 * 3 = 1.5, ceil = 2
        assert_eq!(miner.threshold, 2);
    }

    #[test]
    fn test_threshold_absolute() {
        let mut miner = RustGradualMiner::new(5.0, None, Some(false));
        let transactions = vec![vec![1.0]; 10];

        miner.load_transactions(transactions, 1);

        // Absolute threshold
        assert_eq!(miner.threshold, 5);
    }
}
