# image-text-to-text Architecture Summary (arc-based)

## Overview
- Models analyzed: 8
- Total layers: 3682
- Unique layer types: 64

## Layer Type Distribution
| Layer Type | Count | Percentage |
|-----------|-------|------------|
| Linear | 1357 | 36.9% |
| LayerNorm | 510 | 13.9% |
| Dropout | 466 | 12.7% |
| GELUActivation | 187 | 5.1% |
| BlipTextAttention | 106 | 2.9% |
| BlipTextSelfAttention | 106 | 2.9% |
| BlipTextSelfOutput | 106 | 2.9% |
| BlipMLP | 65 | 1.8% |
| BlipAttention | 65 | 1.8% |
| BlipEncoderLayer | 65 | 1.8% |
| BlipTextIntermediate | 53 | 1.4% |
| BlipTextLayer | 53 | 1.4% |
| BlipTextOutput | 53 | 1.4% |
| Blip2MLP | 39 | 1.1% |
| Blip2EncoderLayer | 39 | 1.1% |
| Blip2Attention | 39 | 1.1% |
| OPTSdpaAttention | 32 | 0.9% |
| ReLU | 32 | 0.9% |
| OPTDecoderLayer | 32 | 0.9% |
| Blip2QFormerSelfOutput | 18 | 0.5% |

## Sample Models

### Taiyi-BLIP-750M-Chinese
- Layers: 955

Layer sequence (first 5):
```
1. BlipTextModel
2. BlipTextEmbeddings
3. Embedding
4. Embedding
5. LayerNorm
```

### cap_v0
- Layers: 114

Layer sequence (first 5):
```
1. Embedding
2. Embedding
3. LayerNorm
4. Dropout
5. GitEmbeddings
```

### blip2-opt-2.7b
- Layers: 1077

Layer sequence (first 5):
```
1. Blip2VisionModel
2. Blip2VisionEmbeddings
3. Conv2d
4. Blip2Encoder
5. ModuleList
```