"""
Main script for analyzing neural network architectures
"""

import os
from config import create_output_dirs, OUTPUT_DIR
from data_loader import load_csv_files
from basic_analysis import (
    analyze_layer_types, analyze_blocks, common_unique_layers,
    compare_layer_distributions, analyze_specialized_components,
    analyze_hierarchy
)
from advanced_analysis import (
    plot_layer_distributions, plot_task_comparison,
    create_architecture_patterns_report, generate_task_summaries,
    export_comparative_data
)

def main():
    """Main function to analyze neural network architectures"""
    print("Neural Network Architecture Analyzer")
    print("===================================")
    
    # Create output directories
    create_output_dirs()
    print(f"Output will be saved to: {OUTPUT_DIR}")
    
    # Load data
    print("\nLoading model architecture data...")
    task_dfs, model_dfs = load_csv_files()
    
    if not task_dfs:
        print("No data found to analyze. Exiting.")
        return
    
    task_count = len(task_dfs)
    model_count = sum(len(models) for models in model_dfs.values())
    print(f"Loaded data for {task_count} tasks and {model_count} models.")
    
    # Basic analysis
    print("\nPerforming basic analysis...")
    layer_type_counts = analyze_layer_types(task_dfs)
    block_counts = analyze_blocks(task_dfs)
    common_layers, unique_layers = common_unique_layers(task_dfs)
    layer_distribution_comparison = compare_layer_distributions(task_dfs)
    specialized_components = analyze_specialized_components(task_dfs)
    hierarchy_patterns = analyze_hierarchy(task_dfs)
    
    # Print some basic results
    print(f"Found {len(common_layers)} common layer types across all tasks:")
    if common_layers:
        print(", ".join(sorted(common_layers)))
    
    # Advanced analysis and visualization
    print("\nGenerating visualizations...")
    plot_layer_distributions(layer_type_counts)
    plot_task_comparison(layer_distribution_comparison)
    
    # Create reports and summaries
    print("\nCreating reports and summaries...")
    create_architecture_patterns_report(task_dfs, hierarchy_patterns, specialized_components)
    generate_task_summaries(task_dfs, model_dfs)
    export_comparative_data(task_dfs)
    
    # Print summary
    print("\nAnalysis complete!")
    print(f"Results saved to {OUTPUT_DIR}")
    print("- Visualizations: images/")
    print("- Reports: reports/")
    print("- Statistics: statistics_txt/")
    print("- Comparative data: csv_data/")

if __name__ == "__main__":
    main() 