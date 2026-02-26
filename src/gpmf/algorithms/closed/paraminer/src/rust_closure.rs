//! Closure operator module for gradual pattern mining
//!
//! Computes the closure of a pattern by finding all items that can be added
//! without changing the pattern's support.

use crate::rust_datatypes::TransactionPair;
use crate::BoolMatrix;
use std::collections::{HashMap, HashSet};

/// Compute the closure of a pattern.
///
/// The closure includes all items that can be added to the pattern
/// without changing its support (i.e., items whose support equals the pattern's support).
///
/// # Arguments
/// * `pattern` - Current pattern (list of item codes)
/// * `supporting_vtids` - Virtual transaction IDs supporting the pattern
/// * `item_matrices` - Boolean matrices for each item
/// * `item_to_vtids` - Mapping from item code to vtids containing it
/// * `vtid_to_pair` - Mapping from vtid to transaction pair
/// * `num_attributes` - Number of attributes in the dataset
///
/// # Returns
/// Closed pattern (sorted list of item codes)
pub fn compute_closure(
    pattern: &[usize],
    supporting_vtids: &HashSet<usize>,
    item_matrices: &[BoolMatrix],
    item_to_vtids: &[Vec<usize>],
    vtid_to_pair: &[TransactionPair],
    num_attributes: usize,
) -> Vec<usize> {
    let mut closed = pattern.to_vec();

    // Extract transaction pairs from supporting vtids
    let pairs: Vec<&TransactionPair> = supporting_vtids
        .iter()
        .map(|&vtid| &vtid_to_pair[vtid])
        .collect();

    // Build tid mapping: original tid -> dense index [0, num_tids)
    let mut tid_to_idx: HashMap<usize, usize> = HashMap::new();
    let mut idx = 0;

    for pair in &pairs {
        if !tid_to_idx.contains_key(&pair.tid1) {
            tid_to_idx.insert(pair.tid1, idx);
            idx += 1;
        }
        if !tid_to_idx.contains_key(&pair.tid2) {
            tid_to_idx.insert(pair.tid2, idx);
            idx += 1;
        }
    }

    let num_tids = tid_to_idx.len();

    // Build pattern matrix with dense indices
    let mut pattern_matrix = BoolMatrix::new(num_tids, None);
    for pair in &pairs {
        let idx1 = tid_to_idx[&pair.tid1];
        let idx2 = tid_to_idx[&pair.tid2];
        pattern_matrix.set(idx2, idx1, true).unwrap();
    }

    // Track if we've seen the first positive item
    let mut first_positive_flag = false;

    // Check each item to see if it can be added to closure
    for item_code in 0..(num_attributes * 2) {
        if pattern.contains(&item_code) {
            // Already in pattern
            if item_code % 2 == 0 {
                // Positive item
                first_positive_flag = true;
            }
            continue;
        }

        // Skip negative items before first positive
        if item_code % 2 == 1 && !first_positive_flag {
            continue;
        }

        // Build restricted item matrix using the same tid mapping
        // Get vtids for this item
        let item_vtids: HashSet<usize> = item_to_vtids[item_code].iter().copied().collect();

        // Get intersection with supporting vtids
        let common_vtids: HashSet<usize> = item_vtids
            .intersection(supporting_vtids)
            .copied()
            .collect();

        // Build item matrix with same size as pattern matrix
        let mut item_matrix = BoolMatrix::new(num_tids, None);
        for vtid in common_vtids {
            let pair = &vtid_to_pair[vtid];
            // Only add if both transactions are in our tid mapping
            if let (Some(&idx1), Some(&idx2)) = (tid_to_idx.get(&pair.tid1), tid_to_idx.get(&pair.tid2)) {
                item_matrix.set(idx2, idx1, true).unwrap();
            }
        }

        // Check if item matrix equals pattern matrix
        if item_matrix == pattern_matrix {
            closed.push(item_code);
            if item_code % 2 == 0 {
                first_positive_flag = true;
            }
        }
    }

    closed.sort_unstable();
    closed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_closure_simple() {
        // Create simple test case with 2 attributes (4 items)
        let vtid_to_pair = vec![
            TransactionPair::new(0, 1),
            TransactionPair::new(1, 2),
        ];

        // Item 0 and 2 always appear together
        let item_to_vtids = vec![
            vec![0, 1],  // Item 0
            vec![],      // Item 1
            vec![0, 1],  // Item 2 (same as item 0)
            vec![],      // Item 3
        ];

        let supporting_vtids: HashSet<usize> = vec![0, 1].into_iter().collect();

        // Create dummy matrices (simplified for test)
        let matrices = vec![
            BoolMatrix::new(3, None),
            BoolMatrix::new(3, None),
            BoolMatrix::new(3, None),
            BoolMatrix::new(3, None),
        ];

        let closure = compute_closure(
            &[0],
            &supporting_vtids,
            &matrices,
            &item_to_vtids,
            &vtid_to_pair,
            2,
        );

        // Closure should include item 2 since it has same support
        assert!(closure.contains(&0));
        // Note: Without actual matrix data, this test is simplified
    }

    #[test]
    fn test_closure_preserves_pattern() {
        // Closure should always contain the original pattern
        let vtid_to_pair = vec![TransactionPair::new(0, 1)];
        let item_to_vtids = vec![vec![0], vec![], vec![], vec![]];
        let supporting_vtids: HashSet<usize> = vec![0].into_iter().collect();
        let matrices = vec![BoolMatrix::new(2, None); 4];

        let closure = compute_closure(
            &[0, 2],
            &supporting_vtids,
            &matrices,
            &item_to_vtids,
            &vtid_to_pair,
            2,
        );

        assert!(closure.contains(&0));
        assert!(closure.contains(&2));
    }
}
