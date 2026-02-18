//! Data conversion module for gradual pattern mining
//!
//! Converts transactions to virtual transactions and builds boolean matrices.

use crate::rust_datatypes::{TransactionPair, VirtualTransaction};
use crate::BoolMatrix;
use rayon::prelude::*;
use std::sync::Mutex;

/// Convert original transactions to virtual transactions with gradual items.
///
/// For each pair of transactions (i, j) where i != j:
/// - If value[k] in trans_i < value[k] in trans_j, add item 2*k (increase)
/// - If value[k] in trans_i > value[k] in trans_j, add item 2*k+1 (decrease)
/// - If equal, add the item based on i>j (handles null variations per GLCM definition)
///
/// # Arguments
/// * `transactions` - List of transaction vectors (each is a Vec<f64>)
/// * `num_attributes` - Number of attributes per transaction
///
/// # Returns
/// Tuple of (virtual_transactions, transaction_pairs)
pub fn convert_to_gradual_items(
    transactions: &[Vec<f64>],
    num_attributes: usize,
) -> (Vec<VirtualTransaction>, Vec<TransactionPair>) {
    let num_transactions = transactions.len();
    let expected_vtrans = num_transactions * (num_transactions - 1);

    // Pre-allocate vectors
    let vtransactions = Mutex::new(Vec::with_capacity(expected_vtrans));
    let pairs = Mutex::new(Vec::with_capacity(expected_vtrans));

    // Process all pairs in parallel
    (0..num_transactions)
        .into_par_iter()
        .for_each(|i| {
            let trans_i = &transactions[i];
            let mut local_vtrans = Vec::new();
            let mut local_pairs = Vec::new();

            for j in 0..num_transactions {
                if i == j {
                    continue;
                }

                let trans_j = &transactions[j];
                let mut items = Vec::with_capacity(num_attributes);

                // Compare each attribute
                for k in 0..num_attributes {
                    let val_i = trans_i[k];
                    let val_j = trans_j[k];

                    let item_code = if val_i < val_j {
                        // Increase: X+
                        2 * k
                    } else if val_i > val_j {
                        // Decrease: X-
                        2 * k + 1
                    } else {
                        // No variation: follow GLCM definition
                        // Add 2*k if i < j, else 2*k+1
                        2 * k + if i > j { 1 } else { 0 }
                    };

                    items.push(item_code);
                }

                let pair = TransactionPair::new(i, j);
                local_pairs.push(pair.clone());

                // vtid will be set later
                local_vtrans.push((items, pair));
            }

            // Add to global collections
            let mut vt = vtransactions.lock().unwrap();
            let mut p = pairs.lock().unwrap();
            p.extend(local_pairs);
            vt.extend(local_vtrans);
        });

    let vtransactions_vec = vtransactions.into_inner().unwrap();
    let pairs = pairs.into_inner().unwrap();

    // Assign vtids sequentially
    let vtransactions: Vec<_> = vtransactions_vec
        .into_iter()
        .enumerate()
        .map(|(vtid, (items, pair))| VirtualTransaction::new(items, pair, vtid))
        .collect();

    (vtransactions, pairs)
}

/// Build boolean matrices for each gradual item.
///
/// For each item, the matrix BM[i,j] = 1 if the virtual transaction
/// formed by original transactions (i,j) contains that item.
///
/// # Arguments
/// * `vtransactions` - List of virtual transactions
/// * `num_items` - Total number of items (2 * num_attributes)
/// * `num_transactions` - Number of original transactions
///
/// # Returns
/// Tuple of (item_matrices, item_to_vtids)
/// - item_matrices: Vec of BoolMatrix, one per item
/// - item_to_vtids: Vec of Vec<usize>, mapping each item to vtids containing it
pub fn build_item_matrices(
    vtransactions: &[VirtualTransaction],
    num_items: usize,
    num_transactions: usize,
) -> (Vec<BoolMatrix>, Vec<Vec<usize>>) {
    // Build transposed representation: item -> list of vtids
    let mut item_to_vtids: Vec<Vec<usize>> = vec![Vec::new(); num_items];

    for vtrans in vtransactions {
        for &item in &vtrans.items {
            item_to_vtids[item].push(vtrans.vtid);
        }
    }

    // Build boolean matrices
    let item_matrices: Vec<BoolMatrix> = (0..num_items)
        .into_par_iter()
        .map(|item_code| {
            let mut matrix = BoolMatrix::new(num_transactions, None);

            for &vtid in &item_to_vtids[item_code] {
                let pair = &vtransactions[vtid].pair;
                // Set matrix[tid2, tid1] = 1 (note the order)
                matrix.set(pair.tid2, pair.tid1, true).unwrap();
            }

            matrix
        })
        .collect();

    (item_matrices, item_to_vtids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_simple() {
        let transactions = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 1.0],
        ];

        let (vtransactions, pairs) = convert_to_gradual_items(&transactions, 2);

        // Should have n * (n-1) virtual transactions
        assert_eq!(vtransactions.len(), 6);
        assert_eq!(pairs.len(), 6);

        // Check first virtual transaction (0, 1)
        // trans[0] = [1.0, 2.0], trans[1] = [2.0, 3.0]
        // Attribute 0: 1.0 < 2.0 -> item 0 (increase)
        // Attribute 1: 2.0 < 3.0 -> item 2 (increase)
        let vt_0_1 = vtransactions.iter().find(|vt| vt.pair.tid1 == 0 && vt.pair.tid2 == 1).unwrap();
        assert_eq!(vt_0_1.items, vec![0, 2]);
    }

    #[test]
    fn test_build_matrices() {
        let transactions = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
        ];

        let (vtransactions, _) = convert_to_gradual_items(&transactions, 2);
        let (matrices, item_to_vtids) = build_item_matrices(&vtransactions, 4, 2);

        // Should have 4 matrices (2 attributes * 2 variations)
        assert_eq!(matrices.len(), 4);
        assert_eq!(item_to_vtids.len(), 4);

        // Check that matrices have correct size
        for matrix in &matrices {
            assert_eq!(matrix.get_size(), 2);
        }
    }
}
