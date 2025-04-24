"""
Script to rename CSV files based on the Model ID field in each file
"""

import os
import pandas as pd
import glob
from pathlib import Path

def rename_csv_files():
    """Rename CSV files using the Model ID field in each file"""
    # Define input directory
    input_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "model_architecture", "raw")
    
    # Count for status reporting
    total_files = 0
    renamed_files = 0
    
    print("Starting to process CSV files...")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        return
    
    # Process each subdirectory (task directory)
    for task_dir in os.listdir(input_dir):
        task_path = os.path.join(input_dir, task_dir)
        
        # Skip if not a directory
        if not os.path.isdir(task_path):
            continue
            
        print(f"Processing task: {task_dir}")
        
        # Find all CSV files in the task directory
        csv_files = glob.glob(os.path.join(task_path, "*.csv"))
        
        for csv_file in csv_files:
            total_files += 1
            
            try:
                # Read the CSV file
                df = pd.read_csv(csv_file)
                
                # Check if "Model ID" column exists
                if "Model ID" not in df.columns:
                    print(f"Warning: 'Model ID' column not found in {csv_file}, skipping")
                    continue
                
                # Get the first non-empty Model ID value
                model_id = None
                for value in df["Model ID"]:
                    if pd.notna(value) and str(value).strip():
                        model_id = str(value)
                        break
                
                if not model_id:
                    print(f"Warning: No valid Model ID found in {csv_file}, skipping")
                    continue
                
                # Extract the part after "/"
                if "/" in model_id:
                    new_name = model_id.split("/", 1)[1]
                    
                    # Sanitize filename - replace special characters
                    new_name = new_name.replace("/", "_")
                    
                    # Create the new file path with the extracted name
                    dir_path = os.path.dirname(csv_file)
                    new_file_path = os.path.join(dir_path, f"{new_name}.csv")
                    
                    # Check if the new filename already exists
                    if os.path.exists(new_file_path) and csv_file != new_file_path:
                        print(f"Warning: Cannot rename {csv_file} to {new_file_path} as it already exists")
                        continue
                    
                    # Rename the file
                    os.rename(csv_file, new_file_path)
                    renamed_files += 1
                    print(f"Renamed: {os.path.basename(csv_file)} -> {os.path.basename(new_file_path)}")
                else:
                    print(f"Warning: Model ID in {csv_file} doesn't contain '/', skipping")
            
            except Exception as e:
                print(f"Error processing {csv_file}: {str(e)}")
    
    print(f"\nSummary: Processed {total_files} files, renamed {renamed_files} files")

if __name__ == "__main__":
    rename_csv_files() 