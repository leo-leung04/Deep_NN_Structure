#!/usr/bin/env python
"""
Model Architecture Deduplication Runner
Allows choosing between different deduplication strategies
"""

import argparse
import sys
import os

def main():
    """Run model architecture deduplication with selected strategy"""
    parser = argparse.ArgumentParser(description="Model Architecture Deduplication Tool")
    
    parser.add_argument(
        "--strategy", 
        type=str, 
        choices=["name", "architecture", "both"], 
        default="name",
        help="Deduplication strategy to use: 'name' (base model name), 'architecture' (model structure), or 'both'"
    )
    
    parser.add_argument(
        "--similarity-threshold", 
        type=float, 
        default=0.85,
        help="Similarity threshold for architecture-based deduplication (0.0-1.0, default: 0.85)"
    )
    
    args = parser.parse_args()
    
    # Import strategies
    try:
        from .deduplicate_with_name import deduplicate_models as name_based_dedup
        from .deduplicate_with_architecture import deduplicate_models as arch_based_dedup
    except ImportError:
        # If running directly
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data_processing.deduplicate_with_name import deduplicate_models as name_based_dedup
        from data_processing.deduplicate_with_architecture import deduplicate_models as arch_based_dedup
    
    # Run selected strategy
    if args.strategy == "name":
        print("Running name-based deduplication...")
        name_based_dedup()
    elif args.strategy == "architecture":
        print(f"Running architecture-based deduplication (similarity threshold: {args.similarity_threshold})...")
        arch_based_dedup(similarity_threshold=args.similarity_threshold)
    elif args.strategy == "both":
        print("Running both deduplication strategies sequentially...")
        print("\n=== Name-based Deduplication ===")
        name_based_dedup()
        print("\n=== Architecture-based Deduplication ===")
        arch_based_dedup(similarity_threshold=args.similarity_threshold)
    
    print("Deduplication complete!")

if __name__ == "__main__":
    main() 