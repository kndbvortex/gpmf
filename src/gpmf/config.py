"""Configuration management for gradual mining package."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging


@dataclass
class GradualMiningConfig:
    """Configuration for gradual mining algorithms.

    Attributes:
        verbose: Enable verbose output
        suppress_prints: Suppress all print statements
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        cache_dir: Directory for caching intermediate results
        use_cache: Enable caching
        n_jobs: Default number of parallel jobs (-1 for all cores)
        spmf_path: Path to SPMF jar file (for seasonal patterns)
    """

    verbose: bool = False
    suppress_prints: bool = True
    log_level: str = "WARNING"
    cache_dir: Optional[Path] = None
    use_cache: bool = False
    n_jobs: int = 1
    spmf_path: Optional[str] = None

    def __post_init__(self):
        """Initialize configuration from environment variables."""
        # Override with environment variables if set
        if os.getenv('GRADUAL_MINING_VERBOSE'):
            self.verbose = os.getenv('GRADUAL_MINING_VERBOSE', '').lower() == 'true'

        if os.getenv('GRADUAL_MINING_LOG_LEVEL'):
            self.log_level = os.getenv('GRADUAL_MINING_LOG_LEVEL', 'WARNING')

        if os.getenv('GRADUAL_MINING_CACHE_DIR'):
            self.cache_dir = Path(os.getenv('GRADUAL_MINING_CACHE_DIR'))

        if os.getenv('GRADUAL_MINING_N_JOBS'):
            self.n_jobs = int(os.getenv('GRADUAL_MINING_N_JOBS', '1'))

        if os.getenv('SPMF_PATH'):
            self.spmf_path = os.getenv('SPMF_PATH')

        # Set up cache directory
        if self.cache_dir is None:
            self.cache_dir = Path.home() / '.cache' / 'gradual_mining'

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Configure logging based on settings."""
        log_level = getattr(logging, self.log_level.upper(), logging.WARNING)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


# Global configuration instance
config = GradualMiningConfig()
