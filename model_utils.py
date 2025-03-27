#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Utilities Module - Responsible for creating models and extracting model architectures
"""

import torch
import torch.nn as nn
import logging
from transformers import (
    AutoConfig, AutoModel, AutoProcessor, AutoTokenizer, 
    AutoModelForImageClassification, AutoModelForObjectDetection,
    AutoModelForImageSegmentation, AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModelForQuestionAnswering,
    AutoModelForMaskedLM, AutoModelForTokenClassification
)

# Get logger
logger = logging.getLogger(__name__)

# Global dictionary to track unique base models
processed_base_models = {}

# -----------------------------
# Function: Create model for specific task
def create_model_for_task(model_id, task_type, base_model_name=None):
    """Create model instance based on task type without downloading weights"""
    # Skip if this base model has already been processed
    if base_model_name is not None and base_model_name in processed_base_models:
        logger.info(f"Skipping {model_id} - Base model {base_model_name} already processed")
        return None
        
    # Mark this base model as processed if provided
    if base_model_name is not None:
        processed_base_models[base_model_name] = model_id
    
    try:
        # Get model config instead of full model
        config = AutoConfig.from_pretrained(model_id)
        
        # Use specialized model loaders based on task type, but using from_config instead of from_pretrained
        if task_type == "image-classification":
            return AutoModelForImageClassification.from_config(config)
        elif task_type == "object-detection":
            return AutoModelForObjectDetection.from_config(config)
        elif task_type == "image-segmentation":
            return AutoModelForImageSegmentation.from_config(config)
        elif task_type == "text-classification":
            return AutoModelForSequenceClassification.from_config(config)
        elif task_type == "text-generation":
            return AutoModelForCausalLM.from_config(config)
        elif task_type in ["translation", "summarization"]:
            return AutoModelForSeq2SeqLM.from_config(config)
        elif task_type == "question-answering":
            return AutoModelForQuestionAnswering.from_config(config)
        elif task_type == "fill-mask":
            return AutoModelForMaskedLM.from_config(config)
        elif task_type == "token-classification":
            return AutoModelForTokenClassification.from_config(config)
        
        # Multimodal models often need special handling
        elif task_type in ["text-to-image", "image-to-text", "visual-question-answering"]:
            try:
                return AutoModel.from_config(config)
            except Exception as e:
                logger.warning(f"Failed to load multimodal model: {model_id} - {e}")
                return None
        
        # Default fallback to AutoModel
        else:
            return AutoModel.from_config(config)
            
    except Exception as e:
        logger.error(f"Failed to create model {model_id}: {e}")
        return None

# -----------------------------
# Function: Infer input shape dynamically
def get_dummy_input(model, task_type, model_id):
    """Attempt to determine correct input shape based on task type"""
    # Comprehensive default shapes for various tasks
    task_shapes = {
        "image-classification": (1, 3, 224, 224),
        "image-segmentation": (1, 3, 224, 224),
        "object-detection": (1, 3, 640, 640),
        "document-question-answering": (1, 3, 224, 224),  # Document image
        "image-to-image": (1, 3, 256, 256),
        "depth-estimation": (1, 3, 384, 384),
        "video-classification": (1, 3, 8, 224, 224),  # (batch, channels, frames, height, width)
        "text-classification": None,  # Will use tokenizer
        "text-generation": None,
        "translation": None,
        "summarization": None,
        "question-answering": None,
        "fill-mask": None,
        "token-classification": None,
        "sentence-similarity": None,
        "feature-extraction": None,
        "text-to-image": None,  # Needs special handling
        "image-to-text": None,  # Needs special handling
        "visual-question-answering": None,  # Needs special handling
        "image-text-retrieval": None,  # Needs special handling
    }
    
    # Check if task has default shape
    if task_type in task_shapes:
        if task_shapes[task_type] is not None:
            return torch.randn(task_shapes[task_type])
        else:
            # For text-based tasks, avoid downloading tokenizer by creating dummy inputs
            if hasattr(model, "config"):
                # Create dummy inputs based on config
                seq_length = 16
                if hasattr(model.config, "max_position_embeddings"):
                    seq_length = min(16, model.config.max_position_embeddings)
                batch_size = 1
                return {"input_ids": torch.randint(100, 200, (batch_size, seq_length))}
            
            # Only as last resort, try to get tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
                inputs = tokenizer("This is a test input", return_tensors="pt")
                return inputs
            except Exception as e:
                logger.warning(f"Could not create tokenizer input for {model_id}: {e}")
                # Fallback to generic input
                return {"input_ids": torch.randint(100, 200, (1, 16))}
    
    # For unknown task types, try to infer from model default config
    if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
        # This might be a transformer model, create dummy inputs
        return {"input_ids": torch.randint(100, 200, (1, 16))}
    
    # For image models with specified input size
    if hasattr(model, "config") and hasattr(model.config, "image_size"):
        input_size = (1, 3, model.config.image_size, model.config.image_size)
        return torch.randn(input_size)
    
    # Try generic default for vision models
    return torch.randn((1, 3, 224, 224))

# -----------------------------
# Function: Extract model architecture details
def extract_model_details(model, task_type, model_id):
    """Extract architecture details from the model"""
    layers_info = []

    def register_hook(module, name):
        def hook(module, input, output):
            # Process input shape
            if input and len(input) > 0:
                if isinstance(input[0], torch.Tensor):
                    input_shape = list(input[0].shape)
                else:
                    input_shape = "Complex Input"
            else:
                input_shape = "Empty Input"
            
            # Process output shape
            if isinstance(output, torch.Tensor):
                output_shape = list(output.shape)
            elif isinstance(output, tuple) and all(isinstance(o, torch.Tensor) for o in output):
                output_shape = [list(o.shape) for o in output]
            else:
                output_shape = "Complex Output"

            # Extract parameters based on layer type
            params = {}
            if isinstance(module, nn.Conv2d):
                params = {"Kernel Size": module.kernel_size, "Stride": module.stride, "Padding": module.padding,
                          "In Channels": module.in_channels, "Out Channels": module.out_channels}
            elif isinstance(module, nn.Conv1d):
                params = {"Kernel Size": module.kernel_size, "Stride": module.stride, "Padding": module.padding,
                          "In Channels": module.in_channels, "Out Channels": module.out_channels}
            elif isinstance(module, nn.Linear):
                params = {"Input Features": module.in_features, "Output Features": module.out_features}
            elif isinstance(module, nn.LayerNorm):
                params = {"Normalized Shape": module.normalized_shape}
            elif isinstance(module, nn.MultiheadAttention):
                params = {"Num Heads": module.num_heads, "Embed Dim": module.embed_dim}
            elif hasattr(module, "in_features") and hasattr(module, "out_features"):
                params = {"Input Features": module.in_features, "Output Features": module.out_features}
            elif hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                params = {"Weight Shape": list(module.weight.shape)}

            # Add to layer info
            layers_info.append({
                "Layer Name": name, 
                "Layer Type": module.__class__.__name__,
                "Input Shape": input_shape, 
                "Output Shape": output_shape, 
                "Parameters": params
            })
        return hook

    # Register hooks for all modules
    hooks = []
    for name, layer in model.named_modules():
        if not isinstance(layer, nn.Sequential) and not name == '':
            hooks.append(layer.register_forward_hook(register_hook(layer, name)))

    # Get appropriate input based on task type
    dummy_input = get_dummy_input(model, task_type, model_id)
    
    if dummy_input is None:
        logger.warning(f"Could not create dummy input for {model_id}")
        # Remove hooks before returning
        for hook in hooks:
            hook.remove()
        return None
    
    # Try to run model forward pass
    try:
        with torch.no_grad():
            if isinstance(dummy_input, dict):
                model(**dummy_input)
            else:
                model(dummy_input)
    except Exception as e:
        logger.error(f"Skipping model {model_id} due to forward pass issue: {e}")
        # Remove hooks before returning
        for hook in hooks:
            hook.remove()
        return None
    
    # Remove hooks after forward pass
    for hook in hooks:
        hook.remove()

    return layers_info 