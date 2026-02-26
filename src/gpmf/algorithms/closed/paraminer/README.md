# ParaMiner - Parallel Gradual Pattern Mining

This directory contains the ParaMiner implementation for mining closed frequent gradual patterns.

## Usage

### Basic Usage (Python only)

```python
from gradual_mining import ParaMiner

# Use pure Python implementation
miner = ParaMiner(min_support=0.5, use_rust=False)
miner.fit(data)
patterns = miner.get_patterns()
```

### With Rust Acceleration (if compiled)

```python
from gradual_mining import ParaMiner, RUST_AVAILABLE

# Check if Rust is available
print(f"Rust acceleration: {RUST_AVAILABLE}")

# Automatically use Rust if available
miner = ParaMiner(min_support=0.5, use_rust=True)
miner.fit(data)
patterns = miner.get_patterns()
```

### Multi-threaded Execution

```python
# Use 4 threads
miner = ParaMiner(min_support=0.5, num_threads=4)
miner.fit(data)
patterns = miner.get_patterns()
```

## Building Rust Extension (Optional)

To enable Rust acceleration for significant performance improvements:

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin
```

### Build and Install

```bash
# Navigate to this directory
cd gradual_mining/algorithms/parallel/paraminer

# Build and install in development mode
maturin develop --release

# Or build a wheel
maturin build --release
```

