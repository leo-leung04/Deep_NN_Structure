"""
Configuration settings for Hugging Face Model Architecture Extractor
"""

import os
import json
import logging

# --- Configuration Settings ---
# Output directory for model architecture files
OUTPUT_DIR = os.environ.get("MODEL_EXTRACT_OUTPUT_DIR", "model_architecture/raw")

# Time delay between API calls
SLEEP_TIME = float(os.environ.get("MODEL_EXTRACT_SLEEP_TIME", "1.0"))

# Minimum number of models required to process a task
MIN_MODELS_THRESHOLD = int(os.environ.get("MODEL_EXTRACT_MIN_MODELS", "300"))

# --- Skip List ---
# Tasks that have already been processed
# Read from environment variable if available, otherwise use default list
def get_completed_tasks():
    """Get list of completed tasks from environment variable or default list"""
    env_tasks = os.environ.get("MODEL_EXTRACT_COMPLETED_TASKS", "")
    if env_tasks:
        try:
            # Environment variable can be a JSON string array or comma-separated list
            if env_tasks.startswith("["):
                return json.loads(env_tasks)
            else:
                return [task.strip() for task in env_tasks.split(",") if task.strip()]
        except json.JSONDecodeError:
            print(f"Warning: Could not parse COMPLETED_TASKS environment variable: {env_tasks}")
    
    # Default completed tasks if environment variable is not set or invalid
    return ["feature-extraction", "fill-mask", "image-classification", 
            "image-feature-extraction", "image-segmantation"]

COMPLETED_TASKS = get_completed_tasks()

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                       "model_extraction.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__) 

# Log configuration settings
logger.info(f"Output directory: {OUTPUT_DIR}")
logger.info(f"Sleep time: {SLEEP_TIME}")
logger.info(f"Minimum models threshold: {MIN_MODELS_THRESHOLD}")
logger.info(f"Completed tasks: {COMPLETED_TASKS}") 