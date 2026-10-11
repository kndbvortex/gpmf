//! Membership oracle module for gradual pattern mining
//!
//! Tests if extending a pattern maintains the minimum support threshold.

use crate::rust_datatypes::{vtidset_and, vtidset_iter, TransactionPair, VtidSet};
use crate::BoolMatrix;

/// Test if base_pattern U {extension} is a frequent pattern.
///
/// # Arguments
/// * `base_pattern` - Current pattern (list of item codes)
/// * `extension` - Item code to add
/// * `base_vtids` - Virtual transaction IDs supporting base_pattern (bitset)
/// * `item_vtids` - For each item code, the bitset of vtids containing it
/// * `vtid_to_pair` - Mapping from vtid to transaction pair
/// * `num_transactions` - Number of original transactions
/// * `threshold` - Minimum support threshold
///
/// # Returns
/// Option<(support, supporting_vtids)> if pattern is frequent, None otherwise
pub fn test_membership(
    base_pattern: &[usize],
    extension: usize,
    base_vtids: &[u64],
    item_vtids: &[VtidSet],
    vtid_to_pair: &[TransactionPair],
    num_transactions: usize,
    threshold: usize,
) -> Option<(usize, VtidSet)> {
    // Validate: first item of the extended pattern must be an increase (even code)
    let first = base_pattern.iter().copied().chain(std::iter::once(extension)).min()?;
    if first % 2 == 1 {
        return None;
    }

    // Supporting vtids: intersection of the base vtids and the vtids of the extension
    let supporting_vtids = vtidset_and(base_vtids, &item_vtids[extension]);

    if supporting_vtids.iter().all(|&w| w == 0) {
        return None;
    }

    // Build tid mapping: original tid -> dense index [0, num_tids)
    let mut tid_to_idx = vec![u32::MAX; num_transactions];
    let mut num_tids = 0u32;
    for vtid in vtidset_iter(&supporting_vtids) {
        let pair = &vtid_to_pair[vtid];
        for tid in [pair.tid1, pair.tid2] {
            if tid_to_idx[tid] == u32::MAX {
                tid_to_idx[tid] = num_tids;
                num_tids += 1;
            }
        }
    }

    // Build boolean matrix with dense indices
    let mut matrix = BoolMatrix::new(num_tids as usize, None);
    for vtid in vtidset_iter(&supporting_vtids) {
        let pair = &vtid_to_pair[vtid];
        let idx1 = tid_to_idx[pair.tid1] as usize;
        let idx2 = tid_to_idx[pair.tid2] as usize;
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
    use crate::rust_datatypes::vtidset_from_ids;

    #[test]
    fn test_membership_oracle() {
        // Create simple test case
        let vtid_to_pair = vec![
            TransactionPair::new(0, 1),
            TransactionPair::new(1, 2),
            TransactionPair::new(0, 2),
        ];

        let item_vtids: Vec<VtidSet> = vec![
            vtidset_from_ids(&[0, 2], 3),    // Item 0 in vtids 0, 2
            vtidset_from_ids(&[1], 3),       // Item 1 in vtid 1
            vtidset_from_ids(&[0, 1, 2], 3), // Item 2 in all vtids
        ];

        let base_vtids = vtidset_from_ids(&[0, 1, 2], 3);

        // Test extending with item 2 (should work)
        let result = test_membership(&[0], 2, &base_vtids, &item_vtids, &vtid_to_pair, 3, 1);

        assert!(result.is_some());
    }

    #[test]
    fn test_invalid_first_item() {
        // First item must be increase (even code)
        let vtid_to_pair = vec![TransactionPair::new(0, 1)];
        let item_vtids = vec![vtidset_from_ids(&[0], 1), vtidset_from_ids(&[0], 1)];
        let base_vtids = vtidset_from_ids(&[0], 1);

        // Start with item 1 (decrease) - should be invalid
        let result = test_membership(&[], 1, &base_vtids, &item_vtids, &vtid_to_pair, 2, 1);

        assert!(result.is_none());
    }
}
