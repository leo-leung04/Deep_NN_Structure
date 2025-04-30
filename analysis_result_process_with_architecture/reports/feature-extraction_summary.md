# feature-extraction Architecture Summary (arc-based)

## Overview
- Models analyzed: 59
- Total layers: 12866
- Unique layer types: 188

## Layer Type Distribution
| Layer Type | Count | Percentage |
|-----------|-------|------------|
| Linear | 4627 | 36.0% |
| Dropout | 1776 | 13.8% |
| LayerNorm | 1390 | 10.8% |
| GELUActivation | 456 | 3.5% |
| NewGELUActivation | 208 | 1.6% |
| T5LayerNorm | 161 | 1.3% |
| Embedding | 145 | 1.1% |
| Conv1D | 144 | 1.1% |
| FunnelLayer | 104 | 0.8% |
| FunnelRelMultiheadAttention | 104 | 0.8% |
| FunnelPositionwiseFFN | 104 | 0.8% |
| BertLayer | 88 | 0.7% |
| BertOutput | 88 | 0.7% |
| BertIntermediate | 88 | 0.7% |
| BertAttention | 88 | 0.7% |
| BertSelfOutput | 88 | 0.7% |
| BertSdpaSelfAttention | 88 | 0.7% |
| T5Block | 78 | 0.6% |
| T5Attention | 78 | 0.6% |
| T5LayerSelfAttention | 78 | 0.6% |

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

### data2vec-text-base
- Layers: 226

Layer sequence (first 5):
```
1. Embedding
2. Embedding
3. Embedding
4. LayerNorm
5. Dropout
```