# fill-mask Architecture Summary (arc-based)

## Overview
- Models analyzed: 83
- Total layers: 18904
- Unique layer types: 187

## Layer Type Distribution
| Layer Type | Count | Percentage |
|-----------|-------|------------|
| Linear | 6642 | 35.1% |
| LayerNorm | 2240 | 11.8% |
| Dropout | 2190 | 11.6% |
| GELUActivation | 1036 | 5.5% |
| BertSdpaSelfAttention | 432 | 2.3% |
| BertSelfOutput | 432 | 2.3% |
| BertAttention | 432 | 2.3% |
| BertIntermediate | 432 | 2.3% |
| BertOutput | 432 | 2.3% |
| BertLayer | 432 | 2.3% |
| Embedding | 230 | 1.2% |
| RobertaSdpaSelfAttention | 198 | 1.0% |
| RobertaLayer | 198 | 1.0% |
| RobertaOutput | 198 | 1.0% |
| RobertaAttention | 198 | 1.0% |
| RobertaSelfOutput | 198 | 1.0% |
| RobertaIntermediate | 198 | 1.0% |
| XLMRobertaAttention | 76 | 0.4% |
| XLMRobertaIntermediate | 76 | 0.4% |
| XLMRobertaOutput | 76 | 0.4% |

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

### custom-legalbert
- Layers: 219

Layer sequence (first 5):
```
1. Embedding
2. Embedding
3. Embedding
4. LayerNorm
5. Dropout
```

### sikuroberta
- Layers: 219

Layer sequence (first 5):
```
1. Embedding
2. Embedding
3. Embedding
4. LayerNorm
5. Dropout
```