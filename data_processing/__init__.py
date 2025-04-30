"""
Data Processing Scripts for Model Architecture Analysis

Scripts for processing, renaming and deduplicating model architecture files.
Supports multiple deduplication strategies:
- Name-based deduplication (deduplicate_with_name.py)
- Architecture-based deduplication (deduplicate_with_architecture.py)
"""

__version__ = "0.1.0"

# Import deduplication functions for easy access
from .deduplicate_with_name import deduplicate_models as deduplicate_by_name
from .deduplicate_with_architecture import deduplicate_models as deduplicate_by_architecture 