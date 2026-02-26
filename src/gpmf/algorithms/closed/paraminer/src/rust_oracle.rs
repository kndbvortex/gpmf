//! Membership oracle module for gradual pattern mining
//!
//! Tests if extending a pattern maintains the minimum support threshold.

use crate::rust_datatypes::TransactionPair;
use crate::BoolMatrix;
use std::collections::{HashMap, HashSet};

/// Test if base_pattern U {extension} is a frequent pattern.
///
/// # Arguments
/// * `base_pattern` - Current pattern (list of item codes)
/// * `extension` - Item code to add
/// * `base_vtids` - Virtual transaction IDs supporting base_pattern
/// * `item_to_vtids` - Mapping from item code to vtids containing it
/// * `vtid_to_pair` - Mapping from vtid to transaction pair
/// * `threshold` - Minimum support threshold
///
/// # Returns
/// Option<(support, supporting_vtids)> if pattern is frequent, None otherwise
pub fn test_membership(
    base_pattern: &[usize],
    extension: usize,
    base_vtids: &HashSet<usize>,
    item_to_vtids: &[Vec<usize>],
    vtid_to_pair: &[TransactionPair],
    threshold: usize,
) -> Option<(usize, HashSet<usize>)> {
    // Build extended pattern
    let mut pattern = base_pattern.to_vec();
    pattern.push(extension);
    pattern.sort_unstable();

    // Validate: first item must be an increase (even code)
    if pattern[0] % 2 == 1 {
        return None;
    }

    // Get supporting vtids for extension
    let ext_vtids: HashSet<usize> = item_to_vtids[extension].iter().copied().collect();

    // Compute intersection
    let supporting_vtids: HashSet<usize> = base_vtids.intersection(&ext_vtids).copied().collect();

    if supporting_vtids.is_empty() {
        return None;
    }

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

    // Build boolean matrix with dense indices
    let mut matrix = BoolMatrix::new(num_tids, None);
    for pair in pairs {
        let idx1 = tid_to_idx[&pair.tid1];
        let idx2 = tid_to_idx[&pair.tid2];
        matrix.set(idx2, idx1, true).unwrap();
    }

    // Check for cycles (shouldn't happen in valid gradual patterns)
    if matrix.detect_short_cycles() {
        eprintln!("Warning: Short cycle detected in boolean matrix!");
        return None;
    }

    // Compute support
    let support = compute_support(&matrix);

    if support >= threshold {
        Some((support, supporting_vtids))
    } else {
        None
    }
}

/// Compute gradual support (longest path length) using internal function
///
/// This is a simplified version for use within the miner
fn compute_support(matrix: &BoolMatrix) -> usize {
    // Use the existing compute_gradual_support from lib.rs
    // For now, we'll import it via the crate
    crate::compute_gradual_support(matrix).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_membership_oracle() {
        // Create simple test case
        let vtid_to_pair = vec![
            TransactionPair::new(0, 1),
            TransactionPair::new(1, 2),
            TransactionPair::new(0, 2),
        ];

        let item_to_vtids = vec![
            vec![0, 2],       // Item 0 in vtids 0, 2
            vec![1],          // Item 1 in vtid 1
            vec![0, 1, 2],    // Item 2 in all vtids
        ];

        let base_vtids: HashSet<usize> = vec![0, 1, 2].into_iter().collect();

        // Test extending with item 2 (should work)
        let result = test_membership(
            &[0],
            2,
            &base_vtids,
            &item_to_vtids,
            &vtid_to_pair,
            1, // threshold
        );

        assert!(result.is_some());
    }

    #[test]
    fn test_invalid_first_item() {
        // First item must be increase (even code)
        let vtid_to_pair = vec![TransactionPair::new(0, 1)];
        let item_to_vtids = vec![vec![0], vec![0]];
        let base_vtids: HashSet<usize> = vec![0].into_iter().collect();

        // Start with item 1 (decrease) - should be invalid
        let result = test_membership(
            &[],
            1,
            &base_vtids,
            &item_to_vtids,
            &vtid_to_pair,
            1,
        );

        assert!(result.is_none());
    }
}
