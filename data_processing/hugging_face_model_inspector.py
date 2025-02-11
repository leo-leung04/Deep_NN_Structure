import timm
import pandas as pd
import torch
import torch.nn as nn

# Load the pre-trained ConViT-Small model
model = timm.create_model("convit_small.fb_in1k", pretrained=True)

# Function to extract model details with correct information
def extract_model_details(model):
    layers_info = []

    def register_hook(module, name):
        def hook(module, input, output):
            input_shape = list(input[0].shape) if input else "Unknown"
            output_shape = list(output.shape) if isinstance(output, torch.Tensor) else "Unknown"

            # Extracting parameters for specific layer types
            params = {}
            if isinstance(module, nn.Conv2d):  # Convolutional Layer
                params = {
                    "Kernel Size": module.kernel_size,
                    "Stride": module.stride,
                    "Padding": module.padding,
                    "In Channels": module.in_channels,
                    "Out Channels": module.out_channels
                }
            elif isinstance(module, nn.Linear):  # Fully Connected Layer
                params = {
                    "Input Features": module.in_features,
                    "Output Features": module.out_features
                }
            elif isinstance(module, nn.LayerNorm):  # Layer Normalization
                params = {"Normalized Shape": module.normalized_shape}
            elif isinstance(module, nn.MultiheadAttention):  # Multihead Attention
                params = {"Num Heads": module.num_heads, "Embed Dim": module.embed_dim}

            # Determine block name properly
            if "patch_embed" in name:
                block = "Patch Embedding"
            elif name.startswith("blocks"):
                block = ".".join(name.split(".")[:2])  # Extracts full 'blocks.0', 'blocks.1', etc.
            elif "head" in name:
                block = "Head (Classification Layer)"
            else:
                block = "Other"

            layers_info.append({
                "Block": block,
                "Layer Name": name,
                "Layer Type": module.__class__.__name__,
                "Input Shape": input_shape,
                "Output Shape": output_shape,
                "Parameters": params  # Store extracted parameters
            })
        return hook

    # Register hooks for all layers
    for name, layer in model.named_modules():
        if not isinstance(layer, nn.Sequential):  # Ignore wrapper layers
            layer.register_forward_hook(register_hook(layer, name))

    # Run a dummy input through the model to capture input/output shapes
    dummy_input = torch.randn(1, 3, 224, 224)  # Standard ImageNet input shape
    model(dummy_input)

    return layers_info

# Extract architecture details with correct block assignment
layers_data = extract_model_details(model)

# Convert to DataFrame
df = pd.DataFrame(layers_data)

# Convert parameter dictionaries to strings for better CSV readability
df["Parameters"] = df["Parameters"].apply(lambda x: str(x))

# Save as CSV file
df.to_csv("convit_model_with_parameters.csv", index=False)


