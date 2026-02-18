//! Rust implementation of performance-critical components for gradual pattern mining
//!
//! This module provides fast implementations of:
//! - Boolean matrix operations
//! - Path length computation for support calculation
//! - Full gradual pattern mining algorithm
//!
//! Uses rayon for parallel processing when beneficial.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;

// New modules for full Rust implementation
mod rust_datatypes;
mod rust_conversion;
mod rust_oracle;
mod rust_closure;
mod rust_miner_engine;
mod rust_miner;

// Re-export main types
pub use rust_datatypes::{GradualItem, GradualPatternResult, Variation};
pub use rust_miner::RustGradualMiner;

/// Boolean square matrix optimized for gradual pattern mining
#[pyclass]
#[derive(Clone, PartialEq)]
pub struct BoolMatrix {
    size: usize,
    data: Vec<bool>,
}

#[pymethods]
impl BoolMatrix {
    /// Create a new boolean matrix of size x size
    #[new]
    #[pyo3(signature = (size, fill=None))]
    fn new(size: usize, fill: Option<bool>) -> Self {
        let fill = fill.unwrap_or(false);
        BoolMatrix {
            size,
            data: vec![fill; size * size],
        }
    }

    /// Get value at [row, col]
    fn get(&self, row: usize, col: usize) -> PyResult<bool> {
        if row >= self.size || col >= self.size {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Index out of bounds",
            ));
        }
        Ok(self.data[row * self.size + col])
    }

    /// Set value at [row, col]
    fn set(&mut self, row: usize, col: usize, value: bool) -> PyResult<()> {
        if row >= self.size || col >= self.size {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Index out of bounds",
            ));
        }
        self.data[row * self.size + col] = value;
        Ok(())
    }

    /// Check if row has no True values
    fn is_null_row(&self, row: usize) -> PyResult<bool> {
        if row >= self.size {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Row index out of bounds",
            ));
        }
        let start = row * self.size;
        let end = (row + 1) * self.size;
        Ok(!self.data[start..end].iter().any(|&x| x))
    }

    /// Compute bitwise AND with another matrix (returns new matrix)
    fn bitwise_and(&self, other: &BoolMatrix) -> PyResult<BoolMatrix> {
        if self.size != other.size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Matrices must be same size",
            ));
        }

        let mut result = BoolMatrix::new(self.size, None);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] && other.data[i];
        }
        Ok(result)
    }

    /// Compute bitwise AND in-place
    fn bitwise_and_inplace(&mut self, other: &BoolMatrix) -> PyResult<()> {
        if self.size != other.size {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Matrices must be same size",
            ));
        }
        for i in 0..self.data.len() {
            self.data[i] = self.data[i] && other.data[i];
        }
        Ok(())
    }

    /// Check equality with another matrix
    fn __eq__(&self, other: &BoolMatrix) -> bool {
        self.size == other.size && self.data == other.data
    }

    /// Create a deep copy
    fn copy(&self) -> BoolMatrix {
        self.clone()
    }

    /// Get the size of the matrix
    fn get_size(&self) -> usize {
        self.size
    }

    /// Detect short cycles (i,j and j,i both true)
    fn detect_short_cycles(&self) -> bool {
        for i in 0..self.size {
            for j in (i + 1)..self.size {
                if self.data[i * self.size + j] && self.data[j * self.size + i] {
                    return true;
                }
            }
        }
        false
    }

    fn __repr__(&self) -> String {
        format!("BoolMatrix(size={})", self.size)
    }
}

/// Compute path lengths for each node in the matrix
///
/// Uses memoized DFS to find the longest path from each transaction node.
#[pyfunction]
fn compute_path_lengths(matrix: &BoolMatrix) -> PyResult<Vec<usize>> {
    let size = matrix.size;
    let mut path_lengths = vec![0; size];

    // Recursive function with memoization
    fn rec_compute(
        trans: usize,
        matrix: &BoolMatrix,
        path_lengths: &mut Vec<usize>,
    ) -> PyResult<()> {
        if path_lengths[trans] != 0 {
            return Ok(());
        }

        // Check if row is null
        if matrix.is_null_row(trans)? {
            path_lengths[trans] = 1;
            return Ok(());
        }

        let mut max_path = 0;
        for j in 0..matrix.size {
            if matrix.get(trans, j)? {
                if path_lengths[j] == 0 {
                    rec_compute(j, matrix, path_lengths)?;
                }
                let current_path = path_lengths[j] + 1;
                if current_path > max_path {
                    max_path = current_path;
                }
            }
        }

        path_lengths[trans] = if max_path > 0 { max_path } else { 1 };
        Ok(())
    }

    for i in 0..size {
        rec_compute(i, matrix, &mut path_lengths)?;
    }

    Ok(path_lengths)
}

/// Compute path lengths in parallel (for large matrices)
///
/// Uses Rayon for parallel computation when beneficial (size > threshold).
#[pyfunction]
#[pyo3(signature = (matrix, num_threads=None))]
fn compute_path_lengths_parallel(
    matrix: &BoolMatrix,
    num_threads: Option<usize>,
) -> PyResult<Vec<usize>> {
    // Set number of threads if specified
    if let Some(threads) = num_threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    }

    let size = matrix.size;

    // For small matrices, use sequential version
    if size < 50 {
        return compute_path_lengths(matrix);
    }

    // Use Arc for thread-safe sharing
    let matrix = Arc::new(matrix.clone());
    let path_lengths = Arc::new(std::sync::Mutex::new(vec![0; size]));

    // Parallel processing
    (0..size)
        .into_par_iter()
        .try_for_each(|i| -> PyResult<()> {
            let local_matrix = Arc::clone(&matrix);
            let local_paths = Arc::clone(&path_lengths);

            fn rec_compute(
                trans: usize,
                matrix: &BoolMatrix,
                path_lengths: &Arc<std::sync::Mutex<Vec<usize>>>,
            ) -> PyResult<()> {
                {
                    let paths = path_lengths.lock().unwrap();
                    if paths[trans] != 0 {
                        return Ok(());
                    }
                }

                if matrix.is_null_row(trans)? {
                    let mut paths = path_lengths.lock().unwrap();
                    paths[trans] = 1;
                    return Ok(());
                }

                let mut max_path = 0;
                for j in 0..matrix.size {
                    if matrix.get(trans, j)? {
                        {
                            let paths = path_lengths.lock().unwrap();
                            if paths[j] == 0 {
                                drop(paths);
                                rec_compute(j, matrix, path_lengths)?;
                            }
                        }

                        let paths = path_lengths.lock().unwrap();
                        let current_path = paths[j] + 1;
                        if current_path > max_path {
                            max_path = current_path;
                        }
                    }
                }

                let mut paths = path_lengths.lock().unwrap();
                paths[trans] = if max_path > 0 { max_path } else { 1 };
                Ok(())
            }

            rec_compute(i, &local_matrix, &local_paths)
        })?;

    let result = Arc::try_unwrap(path_lengths)
        .unwrap()
        .into_inner()
        .unwrap();
    Ok(result)
}

/// Compute gradual support (longest path length)
#[pyfunction]
fn compute_gradual_support(matrix: &BoolMatrix) -> PyResult<usize> {
    let path_lengths = compute_path_lengths(matrix)?;
    Ok(*path_lengths.iter().max().unwrap_or(&0))
}

/// Compute gradual support with parallel processing
#[pyfunction]
#[pyo3(signature = (matrix, num_threads=None))]
fn compute_gradual_support_parallel(
    matrix: &BoolMatrix,
    num_threads: Option<usize>,
) -> PyResult<usize> {
    let path_lengths = compute_path_lengths_parallel(matrix, num_threads)?;
    Ok(*path_lengths.iter().max().unwrap_or(&0))
}

/// Python module for Rust-accelerated gradual pattern mining
#[pymodule]
fn paraminer_gradual_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Existing low-level exports
    m.add_class::<BoolMatrix>()?;
    m.add_function(wrap_pyfunction!(compute_path_lengths, m)?)?;
    m.add_function(wrap_pyfunction!(compute_path_lengths_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(compute_gradual_support, m)?)?;
    m.add_function(wrap_pyfunction!(compute_gradual_support_parallel, m)?)?;

    // New full Rust implementation exports
    m.add_class::<RustGradualMiner>()?;
    m.add_class::<GradualItem>()?;
    m.add_class::<Variation>()?;
    m.add_class::<GradualPatternResult>()?;

    Ok(())
}
