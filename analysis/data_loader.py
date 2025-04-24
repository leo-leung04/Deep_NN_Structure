"""
Data loading functions for model architecture analysis
"""

import os
import pandas as pd
import glob
from .config import INPUT_DIR

def load_csv_files():
    """Load all CSV files from all subdirectories excluding deduplication_results.csv"""
    task_dfs = {}
    model_dfs = {}
    
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory {INPUT_DIR} does not exist.")
        return {}, {}
    
    # Get all directory names (tasks)
    task_dirs = [d for d in os.listdir(INPUT_DIR) 
               if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    if not task_dirs:
        print(f"No task directories found in {INPUT_DIR}")
        return {}, {}
    
    for task in task_dirs:
        task_path = os.path.join(INPUT_DIR, task)
        csv_files = glob.glob(os.path.join(task_path, "*.csv"))
        
        # Filter out deduplication_results.csv
        csv_files = [f for f in csv_files 
                    if os.path.basename(f) != "deduplication_results.csv"]
        
        if not csv_files:
            continue
            
        task_model_dfs = {}
        task_models = []
        
        # Load all CSV files for this task
        for file in csv_files:
            try:
                model_name = os.path.basename(file).replace('.csv', '')
                df = pd.read_csv(file)
                if 'Layer Type' in df.columns and 'Block' in df.columns:
                    df['Model'] = model_name
                    task_model_dfs[model_name] = df
                    task_models.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if task_models:
            # Combine all dataframes for this task
            task_dfs[task] = pd.concat(task_models, ignore_index=True)
            model_dfs[task] = task_model_dfs
    
    return task_dfs, model_dfs 