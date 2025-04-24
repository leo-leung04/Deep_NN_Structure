"""
Model structure extraction functions for Hugging Face Model Architecture Extractor
"""

import torch
import torch.nn as nn
import pandas as pd
from .config import logger

def extract_model_structure(model):
    """Extract model structure with shapes using dummy inputs"""
    layers_info = []
    blocks = {}
    
    # Optimized hook method
    def register_hook(module, name):
        def hook(module, input, output):
            # Process input shape
            input_shape = [list(i.shape) if isinstance(i, torch.Tensor) else "Unknown" for i in input] if input else "Unknown"
            if isinstance(input_shape, list) and len(input_shape) == 1:
                input_shape = input_shape[0]  # Simplify if only one input
                
            # Process output shape
            if isinstance(output, torch.Tensor):
                output_shape = list(output.shape)
            elif isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                output_shape = [list(o.shape) if isinstance(o, torch.Tensor) else "Unknown" for o in output]
                if len(output_shape) == 1:
                    output_shape = output_shape[0]  # Simplify if only one output
            else:
                output_shape = "Unknown"
                
            # Extract layer parameters
            params = {}
            if isinstance(module, nn.Conv2d):
                params = {
                    "Kernel Size": module.kernel_size, 
                    "Stride": module.stride, 
                    "Padding": module.padding,
                    "In Channels": module.in_channels, 
                    "Out Channels": module.out_channels
                }
            elif isinstance(module, nn.Linear):
                params = {
                    "Input Features": module.in_features, 
                    "Output Features": module.out_features
                }
            elif isinstance(module, nn.LayerNorm):
                params = {"Normalized Shape": module.normalized_shape}
            elif isinstance(module, nn.MultiheadAttention):
                params = {
                    "Num Heads": module.num_heads, 
                    "Embed Dim": module.embed_dim
                }
            else:
                # Extract generic parameters
                for param_name in ["in_features", "out_features", "in_channels", "out_channels", 
                                "kernel_size", "stride", "padding", "groups", "dilation",
                                "num_heads", "hidden_size", "intermediate_size", "normalized_shape"]:
                    if hasattr(module, param_name):
                        params[param_name] = getattr(module, param_name)
            
            # Determine module's block
            block_name = "base"
            name_parts = name.split('.')
            
            # Identify block from module path
            for i, part in enumerate(name_parts[:-1]):  
                # Common block naming patterns
                if any(pattern in part.lower() for pattern in ["block", "layer", "encoder", "decoder"]):
                    # If part contains digits, use it as block identifier
                    if any(c.isdigit() for c in part):
                        block_name = part
                        break
                    # If next part is a digit, combine them
                    elif i+1 < len(name_parts)-1 and name_parts[i+1].isdigit():
                        block_name = f"{part}_{name_parts[i+1]}"
                        break
                    else:
                        block_name = part
                        break
            
            # If no block pattern found, use first part as block
            if block_name == "base" and len(name_parts) > 1:
                block_name = name_parts[0]
                
            # Count layers in this block
            if block_name in blocks:
                blocks[block_name] += 1
            else:
                blocks[block_name] = 1
                
            # Add to layer list
            layers_info.append({
                "Layer Name": name,
                "Layer Type": module.__class__.__name__,
                "Input Shape": str(input_shape),
                "Output Shape": str(output_shape),
                "Block": block_name,
                "Parameters": str(params)
            })
            
        return hook
    
    # Register hooks for all modules
    for name, module in model.named_modules():
        if name:  # Skip root module
            module.register_forward_hook(register_hook(module, name))
    
    # Get appropriate input size
    def get_dummy_input(model):
        """Infer appropriate input shape for model"""
        batch_size = 1
        seq_length = 196  # Common sequence length
        
        # Default simple tensor input
        default_input = torch.randn(batch_size, 3, 224, 224)
        
        try:
            if hasattr(model, 'config'):
                # For Transformer models
                if hasattr(model.config, 'hidden_size'):
                    hidden_size = model.config.hidden_size
                else:
                    hidden_size = 768  # Default value
                    
                if hasattr(model.config, 'max_position_embeddings'):
                    seq_length = min(seq_length, model.config.max_position_embeddings)
                    
                # Set different inputs based on model type
                model_type = getattr(model.config, 'model_type', '').lower()
                
                # Create different inputs based on model type
                if 'vit' in model_type or 'vision' in model_type or model.__class__.__name__.lower().startswith(('vit', 'vision')):
                    # Vision models
                    return torch.randn(batch_size, 3, 224, 224)
                elif 'gpt' in model_type or 'causal' in model_type:
                    # Autoregressive text models
                    return torch.randint(0, hidden_size, (batch_size, seq_length))
                elif 'bert' in model_type or 'roberta' in model_type:
                    # Encoder text models
                    return {
                        'input_ids': torch.randint(0, hidden_size, (batch_size, seq_length)),
                        'attention_mask': torch.ones(batch_size, seq_length)
                    }
                else:
                    # Default text input
                    return {
                        'input_ids': torch.randint(0, hidden_size, (batch_size, seq_length)),
                        'attention_mask': torch.ones(batch_size, seq_length)
                    }
        except Exception as e:
            logger.debug(f"Error determining input shape: {e}")
            
        return default_input
    
    # Execute forward pass to trigger hooks
    try:
        dummy_input = get_dummy_input(model)
        with torch.no_grad():
            try:
                model(dummy_input)
            except TypeError as e:
                if "unhashable type: 'slice'" in str(e):
                    logger.warning(f"Slice error detected: {e}, trying tensor list input")
                    try:
                        # Try using plain tensor list as input
                        dummy_input = [torch.randint(0, 1000, (1, 196))]
                        model(dummy_input)
                    except Exception as e3:
                        logger.warning(f"Tensor list input failed: {e3}, trying dictionary with tensor values")
                        try:
                            dummy_input = {'inputs': torch.randint(0, 1000, (1, 196))}
                            model(dummy_input)
                        except Exception as e4:
                            logger.warning(f"All input strategies failed: {e4}")
                else:
                    logger.warning(f"First forward pass failed: {e}, trying alternative input")
                    try:
                        # Try different input format
                        if isinstance(dummy_input, dict):
                            dummy_input = torch.randint(0, 1000, (1, 196))
                        else:
                            dummy_input = {
                                'input_ids': torch.randint(0, 1000, (1, 196)),
                                'attention_mask': torch.ones(1, 196)
                            }
                        model(dummy_input)
                    except Exception as e2:
                        logger.warning(f"Second forward pass also failed: {e2}, shapes may be missing")
            except Exception as e:
                logger.warning(f"First forward pass failed: {e}, trying alternative input")
                try:
                    # Try different input format
                    if isinstance(dummy_input, dict):
                        dummy_input = torch.randint(0, 1000, (1, 196))
                    else:
                        dummy_input = {
                            'input_ids': torch.randint(0, 1000, (1, 196)),
                            'attention_mask': torch.ones(1, 196)
                        }
                    model(dummy_input)
                except Exception as e2:
                    logger.warning(f"Second forward pass also failed: {e2}, shapes may be missing")
    except Exception as e:
        logger.warning(f"Failed to determine shapes: {e}")
    
    # Log block statistics
    logger.info(f"Model contains {len(blocks)} blocks with {len(layers_info)} total layers")
    for block, count in blocks.items():
        logger.info(f"  - Block '{block}': {count} layers")
        
    return pd.DataFrame(layers_info) 