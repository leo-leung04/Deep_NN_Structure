"""
Run the complete Deep NN Structure analysis pipeline
"""

import os
import time
import subprocess
import argparse

def run_stage(cmd, name):
    """Run a pipeline stage with timing and status output"""
    print(f"\n{'=' * 80}")
    print(f"Running stage: {name}")
    print(f"{'=' * 80}")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    end_time = time.time()
    
    if result.returncode != 0:
        print(f"\nError: {name} stage failed with return code {result.returncode}")
        return False
    
    print(f"\n{name} completed in {end_time - start_time:.2f} seconds")
    return True

def main():
    """Run the complete pipeline"""
    parser = argparse.ArgumentParser(description="Run the Deep NN Structure analysis pipeline")
    parser.add_argument("--extract", action="store_true", help="Run model extraction stage")
    parser.add_argument("--process", action="store_true", help="Run data processing stage")
    parser.add_argument("--analyze", action="store_true", help="Run analysis stage")
    parser.add_argument("--all", action="store_true", help="Run all stages")
    
    args = parser.parse_args()
    
    # If no flags are specified, show help
    if not (args.extract or args.process or args.analyze or args.all):
        parser.print_help()
        return
    
    # Run stages based on flags
    stages_to_run = []
    
    if args.all or args.extract:
        stages_to_run.append(("cd model_extract && python run.py", "Model Extraction"))
    
    if args.all or args.process:
        stages_to_run.extend([
            ("cd data_processing && python rename_csv_files.py", "Rename CSV Files"),
            ("cd data_processing && python deduplication.py", "Deduplication")
        ])
    
    if args.all or args.analyze:
        stages_to_run.append(("cd analysis && python run.py", "Architecture Analysis"))
    
    # Run each stage
    for cmd, name in stages_to_run:
        success = run_stage(cmd, name)
        if not success:
            print(f"\nPipeline stopped at stage: {name}")
            return
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main() 