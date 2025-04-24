"""
Configuration settings for Hugging Face Model Architecture Extractor
"""

import os
import logging

# --- Output Directory ---
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         "model_architecture", "raw")

# --- API Settings ---
SLEEP_TIME = 1.0  # API call interval

# --- Task Filtering ---
MIN_MODELS_THRESHOLD = 300  # Only process tasks with more than this many models

# --- Skip List ---
# Tasks that have already been processed
COMPLETED_TASKS = ["feature-extraction", "fill-mask", "image-classification", 
                  "image-feature-extraction", "image-segmantation"]

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