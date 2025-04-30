# image-feature-extraction Architecture Summary (arc-based)

## Overview
- Models analyzed: 6
- Total layers: 1089
- Unique layer types: 43

## Layer Type Distribution
| Layer Type | Count | Percentage |
|-----------|-------|------------|
| Linear | 278 | 25.5% |
| Dropout | 152 | 14.0% |
| Conv1D | 96 | 8.8% |
| LayerNorm | 55 | 5.1% |
| RMSNorm | 50 | 4.6% |
| ImageGPTLayerNorm | 49 | 4.5% |
| GELUActivation | 26 | 2.4% |
| ImageGPTBlock | 24 | 2.2% |
| AIMv2Block | 24 | 2.2% |
| AIMv2SwiGLUFFN | 24 | 2.2% |
| ImageGPTMLP | 24 | 2.2% |
| QuickGELUActivation | 24 | 2.2% |
| ImageGPTAttention | 24 | 2.2% |
| AIMv2Attention | 24 | 2.2% |
| Data2VecVisionDropPath | 22 | 2.0% |
| ViTOutput | 14 | 1.3% |
| ViTLayer | 14 | 1.3% |
| ViTAttention | 14 | 1.3% |
| ViTIntermediate | 14 | 1.3% |
| ViTSelfOutput | 14 | 1.3% |

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

### parkinsons_pred0.1
- Layers: 1

Layer sequence (first 5):
```
1. ZeroPad2d
```