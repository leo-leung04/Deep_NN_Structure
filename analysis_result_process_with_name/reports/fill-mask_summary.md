# fill-mask Architecture Summary (name-based)

## Overview
- Models analyzed: 130
- Total layers: 29145
- Unique layer types: 187

## Layer Type Distribution
| Layer Type | Count | Percentage |
|-----------|-------|------------|
| Linear | 10138 | 34.8% |
| LayerNorm | 3468 | 11.9% |
| Dropout | 3389 | 11.6% |
| GELUActivation | 1625 | 5.6% |
| BertSdpaSelfAttention | 669 | 2.3% |
| BertSelfOutput | 669 | 2.3% |
| BertAttention | 669 | 2.3% |
| BertIntermediate | 669 | 2.3% |
| BertOutput | 669 | 2.3% |
| BertLayer | 669 | 2.3% |
| RobertaIntermediate | 438 | 1.5% |
| RobertaSdpaSelfAttention | 438 | 1.5% |
| RobertaLayer | 438 | 1.5% |
| RobertaOutput | 438 | 1.5% |
| RobertaAttention | 438 | 1.5% |
| RobertaSelfOutput | 438 | 1.5% |
| Embedding | 370 | 1.3% |
| XLMRobertaSdpaSelfAttention | 100 | 0.3% |
| XLMRobertaSelfOutput | 100 | 0.3% |
| XLMRobertaAttention | 100 | 0.3% |

## Sample Models

### muril-base-cased
- Layers: 219

Layer sequence (first 5):
```
1. Embedding
2. Embedding
3. Embedding
4. LayerNorm
5. Dropout
```

### ruRoberta-large
- Layers: 420

Layer sequence (first 5):
```
1. Embedding
2. Embedding
3. Embedding
4. LayerNorm
5. Dropout
```

### graphcodebert-base
- Layers: 216

Layer sequence (first 5):
```
1. Embedding
2. Embedding
3. Embedding
4. LayerNorm
5. Dropout
```