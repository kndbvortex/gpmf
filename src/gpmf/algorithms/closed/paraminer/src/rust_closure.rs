//! Closure operator module for gradual pattern mining
//!
//! Computes the closure of a pattern by finding all items that can be added
//! without changing the pattern's support.

use crate::rust_datatypes::{vtidset_is_subset, VtidSet};

/// Compute the closure of a pattern.
///
/// The closure includes all items that can be added to the pattern without changing its set of
/// supporting transaction pairs. Each vtid stands for one ordered pair of transactions, so the
/// matrix of the pattern restricted to an item equals the matrix of the pattern exactly when the
/// item contains every supporting vtid.
///
/// # Arguments
/// * `pattern` - Current pattern (list of item codes)
/// * `supporting_vtids` - Virtual transaction IDs supporting the pattern (bitset)
/// * `item_vtids` - For each item code, the bitset of vtids containing it
/// * `num_attributes` - Number of attributes in the dataset
///
/// # Returns
/// Closed pattern (sorted list of item codes)
pub fn compute_closure(
    pattern: &[usize],
    supporting_vtids: &[u64],
    item_vtids: &[VtidSet],
    num_attributes: usize,
) -> Vec<usize> {
    let mut closed = pattern.to_vec();

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

        // Same supporting pairs with the item as without it
        if vtidset_is_subset(supporting_vtids, &item_vtids[item_code]) {
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
    use crate::rust_datatypes::vtidset_from_ids;

    #[test]
    fn test_closure_simple() {
        // 2 attributes (4 items); items 0 and 2 always appear together
        let item_vtids = vec![
            vtidset_from_ids(&[0, 1], 2), // Item 0
            vtidset_from_ids(&[], 2),     // Item 1
            vtidset_from_ids(&[0, 1], 2), // Item 2 (same as item 0)
            vtidset_from_ids(&[], 2),     // Item 3
        ];
        let supporting_vtids = vtidset_from_ids(&[0, 1], 2);

        let closure = compute_closure(&[0], &supporting_vtids, &item_vtids, 2);

        assert_eq!(closure, vec![0, 2]);
    }

    #[test]
    fn test_closure_preserves_pattern() {
        // Closure should always contain the original pattern
        let item_vtids = vec![
            vtidset_from_ids(&[0], 1),
            vtidset_from_ids(&[], 1),
            vtidset_from_ids(&[], 1),
            vtidset_from_ids(&[], 1),
        ];
        let supporting_vtids = vtidset_from_ids(&[0], 1);

        let closure = compute_closure(&[0, 2], &supporting_vtids, &item_vtids, 2);

        assert!(closure.contains(&0));
        assert!(closure.contains(&2));
    }
}
