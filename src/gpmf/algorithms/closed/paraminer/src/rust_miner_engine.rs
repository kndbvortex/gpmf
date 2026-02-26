//! Mining engine module for gradual pattern mining
//!
//! Implements DFS enumeration with closure-based deduplication.

use crate::rust_closure::compute_closure;
use crate::rust_datatypes::{codes_to_items, GradualPatternResult, TransactionPair};
use crate::rust_oracle::test_membership;
use crate::BoolMatrix;
use std::collections::HashSet;

/// Context passed to recursive mining function
pub struct MiningContext<'a> {
    pub item_matrices: &'a [BoolMatrix],
    pub item_to_vtids: &'a [Vec<usize>],
    pub vtid_to_pair: &'a [TransactionPair],
    pub num_attributes: usize,
    pub threshold: usize,
    pub verbose: bool,
}

/// Mine gradual patterns recursively using DFS.
///
/// # Arguments
/// * `current_pattern` - Current pattern being extended
/// * `current_vtids` - Virtual transactions supporting current pattern
/// * `exclusion_list` - Items that should not be considered for extension
/// * `context` - Mining context with matrices and parameters
/// * `found_patterns` - Set of already found pattern codes (for deduplication)
/// * `results` - Vector to store found patterns
/// * `depth` - Current recursion depth (for debugging)
pub fn mine_recursive(
    current_pattern: Vec<usize>,
    current_vtids: HashSet<usize>,
    exclusion_list: HashSet<usize>,
    context: &MiningContext,
    found_patterns: &mut HashSet<Vec<usize>>,
    results: &mut Vec<GradualPatternResult>,
    depth: usize,
) {
    let max_item = context.num_attributes * 2;

    // Try to extend pattern with each possible item
    for item_code in 0..max_item {
        if exclusion_list.contains(&item_code) || current_pattern.contains(&item_code) {
            continue;
        }

        // Test membership
        let membership_result = test_membership(
            &current_pattern,
            item_code,
            &current_vtids,
            context.item_to_vtids,
            context.vtid_to_pair,
            context.threshold,
        );

        if let Some((support, extended_vtids)) = membership_result {
            // Pattern is frequent!
            let mut extended_pattern = current_pattern.clone();
            extended_pattern.push(item_code);
            extended_pattern.sort_unstable();

            if context.verbose {
                println!(
                    "{}Testing {:?} + {}: support={} (threshold={})",
                    "  ".repeat(depth),
                    current_pattern,
                    item_code,
                    support,
                    context.threshold
                );
            }

            // Compute closure
            let closed_pattern = compute_closure(
                &extended_pattern,
                &extended_vtids,
                context.item_matrices,
                context.item_to_vtids,
                context.vtid_to_pair,
                context.num_attributes,
            );

            // Three-level deduplication:

            // 1. First-parent check: If closure added elements smaller than current item_code,
            //    this pattern should have been found from that smaller item
            let new_elements: HashSet<usize> = closed_pattern
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .difference(&extended_pattern.iter().copied().collect())
                .copied()
                .collect();

            if new_elements.iter().any(|&e| e < item_code) {
                if context.verbose {
                    println!(
                        "{}Skipping {:?} (should be found from smaller item)",
                        "  ".repeat(depth),
                        closed_pattern
                    );
                }
                continue;
            }

            // 2. Exclusion check: If any new element is already in exclusion list
            if new_elements.iter().any(|e| exclusion_list.contains(e)) {
                if context.verbose {
                    println!(
                        "{}Skipping {:?} (already explored)",
                        "  ".repeat(depth),
                        closed_pattern
                    );
                }
                continue;
            }

            // 3. Global check: Check if we've already found this exact closed pattern
            if found_patterns.contains(&closed_pattern) {
                if context.verbose {
                    println!(
                        "{}Skipping {:?} (duplicate)",
                        "  ".repeat(depth),
                        closed_pattern
                    );
                }
                continue;
            }

            // Mark this pattern as found
            found_patterns.insert(closed_pattern.clone());

            // Create result
            let items = codes_to_items(&closed_pattern);
            let result = GradualPatternResult::new(items, support);

            if context.verbose {
                println!("{}Found: {}", "  ".repeat(depth), result.__str__());
            }

            results.push(result);

            // Continue mining with CLOSED pattern
            let mut new_exclusion = exclusion_list.clone();
            new_exclusion.extend(closed_pattern.iter().copied());

            mine_recursive(
                closed_pattern,
                extended_vtids,
                new_exclusion,
                context,
                found_patterns,
                results,
                depth + 1,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deduplication() {
        // Test that found_patterns prevents duplicates
        let mut found_patterns = HashSet::new();
        let pattern = vec![0, 2, 4];

        assert!(!found_patterns.contains(&pattern));
        found_patterns.insert(pattern.clone());
        assert!(found_patterns.contains(&pattern));
    }

    #[test]
    fn test_exclusion_check() {
        let exclusion = vec![1, 3, 5].into_iter().collect::<HashSet<_>>();
        let new_elements = vec![2, 3].into_iter().collect::<HashSet<_>>();

        // Should detect that 3 is in exclusion list
        let has_excluded = new_elements.iter().any(|e| exclusion.contains(e));
        assert!(has_excluded);
    }

    #[test]
    fn test_first_parent_check() {
        let item_code = 5;
        let new_elements = vec![2, 6, 8];

        // Should detect that 2 < 5
        let should_skip = new_elements.iter().any(|&e| e < item_code);
        assert!(should_skip);
    }
}
