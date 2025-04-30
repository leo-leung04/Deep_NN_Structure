# image-feature-extraction Architecture Summary (name-based)

## Overview
- Models analyzed: 15
- Total layers: 2836
- Unique layer types: 43

## Layer Type Distribution
| Layer Type | Count | Percentage |
|-----------|-------|------------|
| Linear | 875 | 30.9% |
| Dropout | 357 | 12.6% |
| LayerNorm | 260 | 9.2% |
| GELUActivation | 124 | 4.4% |
| ViTIntermediate | 112 | 3.9% |
| ViTLayer | 112 | 3.9% |
| ViTOutput | 112 | 3.9% |
| ViTAttention | 112 | 3.9% |
| ViTSelfOutput | 112 | 3.9% |
| ViTSelfAttention | 112 | 3.9% |
| Conv1D | 96 | 3.4% |
| RMSNorm | 50 | 1.8% |
| ImageGPTLayerNorm | 49 | 1.7% |
| ImageGPTAttention | 24 | 0.8% |
| AIMv2Block | 24 | 0.8% |
| AIMv2Attention | 24 | 0.8% |
| ImageGPTBlock | 24 | 0.8% |
| ImageGPTMLP | 24 | 0.8% |
| QuickGELUActivation | 24 | 0.8% |
| AIMv2SwiGLUFFN | 24 | 0.8% |

## Sample Models

### imagegpt-small
- Layers: 316

Layer sequence (first 5):
```
1. Embedding
2. Embedding
3. Dropout
4. ImageGPTLayerNorm
5. Conv1D
```

### tiny-vit-random
- Layers: 43

Layer sequence (first 5):
```
1. Conv2d
2. ViTPatchEmbeddings
3. Dropout
4. ViTEmbeddings
5. LayerNorm
```

### EfficientNet_ParkinsonsPred
- Layers: 43

Layer sequence (first 5):
```
1. Conv2d
2. ViTPatchEmbeddings
3. Dropout
4. ViTEmbeddings
5. LayerNorm
```