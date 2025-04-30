# image-text-to-text Architecture Summary (name-based)

## Overview
- Models analyzed: 41
- Total layers: 15782
- Unique layer types: 64

## Layer Type Distribution
| Layer Type | Count | Percentage |
|-----------|-------|------------|
| Linear | 5563 | 35.2% |
| Dropout | 2293 | 14.5% |
| LayerNorm | 2041 | 12.9% |
| GELUActivation | 793 | 5.0% |
| BlipTextSelfOutput | 634 | 4.0% |
| BlipTextSelfAttention | 634 | 4.0% |
| BlipTextAttention | 634 | 4.0% |
| BlipAttention | 341 | 2.2% |
| BlipEncoderLayer | 341 | 2.2% |
| BlipMLP | 341 | 2.2% |
| BlipTextOutput | 317 | 2.0% |
| BlipTextIntermediate | 317 | 2.0% |
| BlipTextLayer | 317 | 2.0% |
| Embedding | 79 | 0.5% |
| GitLayer | 72 | 0.5% |
| GitOutput | 72 | 0.5% |
| GitIntermediate | 72 | 0.5% |
| GitAttention | 72 | 0.5% |
| GitSelfOutput | 72 | 0.5% |
| GitSelfAttention | 72 | 0.5% |

## Sample Models

### Imagecap1
- Layers: 114

Layer sequence (first 5):
```
1. Embedding
2. Embedding
3. LayerNorm
4. Dropout
5. GitEmbeddings
```

### blip_10k_deduped_10epoch_6batch_1e-05lr_AdamW_1e-2wd
- Layers: 487

Layer sequence (first 5):
```
1. BlipTextModel
2. BlipTextEmbeddings
3. Embedding
4. Embedding
5. LayerNorm
```

### captionary-BLIP
- Layers: 487

Layer sequence (first 5):
```
1. BlipTextModel
2. BlipTextEmbeddings
3. Embedding
4. Embedding
5. LayerNorm
```