# feature-extraction Architecture Summary (name-based)

## Overview
- Models analyzed: 208
- Total layers: 43765
- Unique layer types: 188

## Layer Type Distribution
| Layer Type | Count | Percentage |
|-----------|-------|------------|
| Linear | 15259 | 34.9% |
| Dropout | 5593 | 12.8% |
| LayerNorm | 4916 | 11.2% |
| GELUActivation | 2102 | 4.8% |
| BertSelfOutput | 1080 | 2.5% |
| BertSdpaSelfAttention | 1080 | 2.5% |
| BertAttention | 1080 | 2.5% |
| BertIntermediate | 1080 | 2.5% |
| BertOutput | 1080 | 2.5% |
| BertLayer | 1080 | 2.5% |
| Embedding | 565 | 1.3% |
| RobertaSdpaSelfAttention | 360 | 0.8% |
| RobertaAttention | 360 | 0.8% |
| RobertaIntermediate | 360 | 0.8% |
| RobertaOutput | 360 | 0.8% |
| RobertaLayer | 360 | 0.8% |
| RobertaSelfOutput | 360 | 0.8% |
| T5LayerNorm | 309 | 0.7% |
| NewGELUActivation | 250 | 0.6% |
| Conv1D | 192 | 0.4% |

## Sample Models

### small-base
- Layers: 199

Layer sequence (first 5):
```
1. Embedding
2. LayerNorm
3. Dropout
4. FunnelEmbeddings
5. Dropout
```

### ko-sroberta-multitask
- Layers: 214

Layer sequence (first 5):
```
1. Embedding
2. Embedding
3. Embedding
4. LayerNorm
5. Dropout
```

### koelectra-base-v3-cola-finetuned
- Layers: 223

Layer sequence (first 5):
```
1. Embedding
2. Embedding
3. Embedding
4. LayerNorm
5. Dropout
```