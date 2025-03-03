"""
Utility modules for the S-NAS library.

This package contains utility functions and classes used throughout S-NAS.
"""

# Import modules to make them available from the package
from .job_distributor import JobDistributor, ParallelEvaluator
from .exceptions import (
    SNASException, 
    EvaluationError, 
    ArchitectureError,
    ValidationError,
    PersistenceError
)