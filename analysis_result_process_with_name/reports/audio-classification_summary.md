# audio-classification Architecture Summary (name-based)

## Overview
- Models analyzed: 162
- Total layers: 36330
- Unique layer types: 82

## Layer Type Distribution
| Layer Type | Count | Percentage |
|-----------|-------|------------|
| Linear | 10717 | 29.5% |
| Dropout | 8587 | 23.6% |
| GELUActivation | 5495 | 15.1% |
| LayerNorm | 3611 | 9.9% |
| Conv1d | 1264 | 3.5% |
| _WeightNorm | 948 | 2.6% |
| ParametrizationList | 948 | 2.6% |
| Wav2Vec2FeedForward | 924 | 2.5% |
| Wav2Vec2NoLayerNormConvLayer | 462 | 1.3% |
| WhisperSdpaAttention | 276 | 0.8% |
| Wav2Vec2LayerNormConvLayer | 232 | 0.6% |
| ASTLayer | 221 | 0.6% |
| ASTSelfAttention | 221 | 0.6% |
| ASTSelfOutput | 221 | 0.6% |
| ASTIntermediate | 221 | 0.6% |
| ASTOutput | 221 | 0.6% |
| ASTAttention | 221 | 0.6% |
| Wav2Vec2SamePadLayer | 107 | 0.3% |
| Wav2Vec2FeatureEncoder | 106 | 0.3% |
| HubertNoLayerNormConvLayer | 102 | 0.3% |

## Sample Models

### wav2vec2_spoof_dection1-finetuned-spoofing-classifier
- Layers: 255

Layer sequence (first 5):
```
1. Conv1d
2. GELUActivation
3. Conv1d
4. GELUActivation
5. GELUActivation
```

### data2vec-audio-base-960h-finetuned-gtzan
- Layers: 263

Layer sequence (first 5):
```
1. Conv1d
2. GELUActivation
3. Conv1d
4. LayerNorm
5. GELUActivation
```

### soundclassification
- Layers: 255

Layer sequence (first 5):
```
1. Conv1d
2. GELUActivation
3. Conv1d
4. GELUActivation
5. GELUActivation
```