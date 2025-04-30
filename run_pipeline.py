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
    parser.add_argument("--dedup-strategy", choices=["name", "architecture", "both"], default="name",
                        help="Deduplication strategy to use: name-based, architecture-based, or both (default: name)")
    parser.add_argument("--similarity-threshold", type=float, default=0.85,
                        help="Similarity threshold for architecture-based deduplication (0.0-1.0, default: 0.85)")
    
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
        stages_to_run.append(("cd data_processing && python rename_csv_files.py", "Rename CSV Files"))
        dedup_cmd = f"cd data_processing && python deduplication_runner.py --strategy {args.dedup_strategy}"
        if args.dedup_strategy in ["architecture", "both"]:
            dedup_cmd += f" --similarity-threshold {args.similarity_threshold}"
        stages_to_run.append((dedup_cmd, f"Deduplication ({args.dedup_strategy}-based)"))
    
    if args.all or args.analyze:
        # Update the model architecture analyzer command to use the appropriate directory
        analysis_cmd = "python model_architecture_analyzer.py"
        if args.dedup_strategy == "name":
            analysis_cmd += " --input_dir model_architecture/process_with_name"
        elif args.dedup_strategy == "architecture":
            analysis_cmd += " --input_dir model_architecture/process_with_architecture"
        # Default to name-based if both are used
        else:
            analysis_cmd += " --input_dir model_architecture/process_with_name"
        
        stages_to_run.append((analysis_cmd, "Architecture Analysis"))
    
    # Run each stage
    for cmd, name in stages_to_run:
        success = run_stage(cmd, name)
        if not success:
            print(f"\nPipeline stopped at stage: {name}")
            return
    
    print("\nPipeline completed successfully!")
    
    # Print information about where results are stored
    if args.all or args.analyze:
        print("\nResults are stored in:")
        if args.dedup_strategy == "name" or args.dedup_strategy == "both":
            print("  - Name-based deduplication: model_architecture/process_with_name/")
        if args.dedup_strategy == "architecture" or args.dedup_strategy == "both":
            print("  - Architecture-based deduplication: model_architecture/process_with_architecture/")
        print("  - Analysis results: analysis_result/")

if __name__ == "__main__":
    main() 