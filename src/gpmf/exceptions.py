"""Custom exceptions for the gradual mining package."""


class GradualMiningError(Exception):
    """Base exception for all gradual mining errors."""
    pass


class InvalidDataError(GradualMiningError):
    """Raised when input data is invalid or malformed."""
    pass


class InvalidAlgorithmError(GradualMiningError):
    """Raised when an unknown algorithm is requested."""
    pass


class InvalidParameterError(GradualMiningError):
    """Raised when invalid parameters are provided."""
    pass


class MiningError(GradualMiningError):
    """Raised when an error occurs during pattern mining."""
    pass


class NotFittedError(GradualMiningError):
    """Raised when trying to get results before fitting."""
    pass
