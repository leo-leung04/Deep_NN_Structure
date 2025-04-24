"""
Deduplicate model architecture files by identifying unique base models
"""

import os
import pandas as pd
import hashlib
import shutil
from collections import defaultdict

def get_architecture_hash(df):
    """Calculate a hash of the model architecture for record-keeping"""
    arch_str = ""
    for _, row in df.iterrows():
        arch_str += f"{row['Layer Name']}|{row['Layer Type']}|{row['Input Shape']}|{row['Output Shape']}|{row.get('Parameters', '')}|{row.get('Block', '')}\n"
    
    return hashlib.md5(arch_str.encode()).hexdigest()[:8]  # Only return first 8 digits for brevity

def deduplicate_models():
    """Deduplicate model architecture files based on base model name"""
    # Define input and output directories
    input_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "model_architecture", "raw")
    output_base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "model_architecture", "processed")
    
    # Check if input directory exists
    if not os.path.exists(input_base_dir):
        print(f"Error: Input directory {input_base_dir} does not exist.")
        return
        
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)

    # Get all task directories
    task_dirs = [d for d in os.listdir(input_base_dir) 
                if os.path.isdir(os.path.join(input_base_dir, d))]
                
    if not task_dirs:
        print(f"No task directories found in {input_base_dir}")
        return

    # Process each task directory
    for task_dir in task_dirs:
        print(f"Processing task: {task_dir}")
        
        # Create output directory for this task
        output_dir = os.path.join(output_base_dir, task_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Input directory for this task
        input_dir = os.path.join(input_base_dir, task_dir)
        
        # Get all CSV files in the task directory
        csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"  No CSV files found in {task_dir}, skipping")
            continue
        
        print(f"  Found {len(csv_files)} CSV files")
        
        # Store base model names and corresponding files
        base_model_files = defaultdict(list)
        
        # Process each CSV file
        for csv_file in csv_files:
            file_path = os.path.join(input_dir, csv_file)
            
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Extract model name (remove .csv extension)
                model_name = os.path.splitext(csv_file)[0]
                
                # Try to extract Base Model Name from CSV
                if 'Base Model Name' in df.columns:
                    base_models = df['Base Model Name'].unique()
                    if len(base_models) > 0 and str(base_models[0]) != 'nan':
                        base_model_name = base_models[0]
                    else:
                        base_model_name = model_name
                else:
                    base_model_name = model_name
                
                # Clean and normalize model name, remove version numbers and special markers
                clean_name = base_model_name.lower()
                # Remove version numbers like v1, v2, etc.
                for suffix in ['-v1', '-v2', '-v3', 'v1', 'v2', 'v3', '-base', '-large', '-small']:
                    if clean_name.endswith(suffix):
                        clean_name = clean_name.replace(suffix, '')
                
                # Store base model name and corresponding file
                base_model_files[clean_name].append({
                    'file': csv_file,
                    'original_name': model_name,
                    'base_model': base_model_name,
                    'architecture_hash': get_architecture_hash(df)  # Calculate architecture hash for info display
                })
                
            except Exception as e:
                print(f"  Error processing {csv_file}: {e}")
        
        # Output deduplication results
        unique_model_count = len(base_model_files)
        total_file_count = len(csv_files)
        print(f"  Found {unique_model_count} unique model names out of {total_file_count} files")
        
        # Create deduplication results CSV
        results = []
        
        # Choose a representative file for each base model name
        for base_name, files_info in base_model_files.items():
            # If only one file, use it directly
            if len(files_info) == 1:
                representative_file = files_info[0]['file']
            else:
                # If multiple files, select based on heuristics
                # For example, prefer files with names closest to base_name
                files_info.sort(key=lambda x: len(x['original_name']))  # Simple heuristic: choose shortest name
                representative_file = files_info[0]['file']
            
            # Copy the representative file
            shutil.copy(
                os.path.join(input_dir, representative_file),
                os.path.join(output_dir, representative_file)
            )
            
            # Record all files with the same base name
            all_files = [info['file'] for info in files_info]
            
            # Record results
            results.append({
                'Base Model Name': base_name,
                'Representative File': representative_file,
                'Duplicate Count': len(files_info),
                'Duplicate Files': ', '.join(all_files),
                'Architecture Hashes': ', '.join([info['architecture_hash'] for info in files_info])
            })
        
        # Save deduplication results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, 'deduplication_results.csv'), index=False)
        print(f"  Saved deduplicated files and results to {output_dir}")

    print("Deduplication by model name completed!")

if __name__ == "__main__":
    deduplicate_models() 