#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration Module - Contains predefined task lists and logging settings
"""

import logging
import os

# -----------------------------
# Logging setup
def setup_logging():
    """Set up logging configuration"""
    # Create log file output directory
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console output handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File output handler
    log_file = os.path.join(logs_dir, 'model_extraction.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger

# -----------------------------
# Predefined task list (if web scraping fails)
FALLBACK_TASKS = {
    # Computer Vision
    "image-classification": {"name": "Image Classification", "category": "Computer Vision"},
    "object-detection": {"name": "Object Detection", "category": "Computer Vision"},
    "image-segmentation": {"name": "Image Segmentation", "category": "Computer Vision"},
    "depth-estimation": {"name": "Depth Estimation", "category": "Computer Vision"},
    "image-to-image": {"name": "Image-to-Image", "category": "Computer Vision"},
    "unconditional-image-generation": {"name": "Image Generation", "category": "Computer Vision"},
    "video-classification": {"name": "Video Classification", "category": "Computer Vision"},
    "zero-shot-image-classification": {"name": "Zero-Shot Image Classification", "category": "Computer Vision"},
    
    # Natural Language Processing
    "text-classification": {"name": "Text Classification", "category": "Natural Language Processing"},
    "token-classification": {"name": "Token Classification", "category": "Natural Language Processing"},
    "table-question-answering": {"name": "Table Question Answering", "category": "Natural Language Processing"},
    "question-answering": {"name": "Question Answering", "category": "Natural Language Processing"},
    "zero-shot-classification": {"name": "Zero-Shot Classification", "category": "Natural Language Processing"},
    "translation": {"name": "Translation", "category": "Natural Language Processing"},
    "summarization": {"name": "Summarization", "category": "Natural Language Processing"},
    "conversational": {"name": "Conversational", "category": "Natural Language Processing"},
    "text-generation": {"name": "Text Generation", "category": "Natural Language Processing"},
    "text2text-generation": {"name": "Text-to-Text Generation", "category": "Natural Language Processing"},
    "fill-mask": {"name": "Fill Mask", "category": "Natural Language Processing"},
    "sentence-similarity": {"name": "Sentence Similarity", "category": "Natural Language Processing"},

    # Audio
    "automatic-speech-recognition": {"name": "Speech Recognition", "category": "Audio"},
    "audio-classification": {"name": "Audio Classification", "category": "Audio"},
    "text-to-speech": {"name": "Text-to-Speech", "category": "Audio"},
    "audio-to-audio": {"name": "Audio-to-Audio", "category": "Audio"},
    
    # Multimodal
    "text-to-image": {"name": "Text-to-Image", "category": "Multimodal"},
    "image-to-text": {"name": "Image-to-Text", "category": "Multimodal"},
    "visual-question-answering": {"name": "Visual Question Answering", "category": "Multimodal"},
    "document-question-answering": {"name": "Document Question Answering", "category": "Multimodal"},
    "image-text-retrieval": {"name": "Image-Text Retrieval", "category": "Multimodal"},
    
    # Tabular Data
    "tabular-classification": {"name": "Tabular Classification", "category": "Tabular Data"},
    "tabular-regression": {"name": "Tabular Regression", "category": "Tabular Data"},
    
    # Reinforcement Learning
    "reinforcement-learning": {"name": "Reinforcement Learning", "category": "Reinforcement Learning"},
} 