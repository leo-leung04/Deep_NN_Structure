# audio-classification Architecture Summary (arc-based)

## Overview
- Models analyzed: 27
- Total layers: 7058
- Unique layer types: 82

## Layer Type Distribution
| Layer Type | Count | Percentage |
|-----------|-------|------------|
| Linear | 2103 | 29.8% |
| Dropout | 1497 | 21.2% |
| GELUActivation | 930 | 13.2% |
| LayerNorm | 779 | 11.0% |
| Conv1d | 209 | 3.0% |
| WhisperSdpaAttention | 204 | 2.9% |
| SiLU | 136 | 1.9% |
| _WeightNorm | 128 | 1.8% |
| ParametrizationList | 128 | 1.8% |
| WhisperEncoderLayer | 68 | 1.0% |
| WhisperDecoderLayer | 68 | 1.0% |
| StableDropout | 48 | 0.7% |
| ReLU | 48 | 0.7% |
| Wav2Vec2LayerNormConvLayer | 40 | 0.6% |
| Wav2Vec2ConformerFeedForward | 37 | 0.5% |
| GLU | 36 | 0.5% |
| MambaRMSNorm | 33 | 0.5% |
| MambaMixer | 32 | 0.5% |
| MambaBlock | 32 | 0.5% |
| Wav2Vec2FeedForward | 24 | 0.3% |

## Sample Models

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

### voice_emo
- Layers: 107

Layer sequence (first 5):
```
1. Conv1d
2. GELUActivation
3. Conv1d
4. LayerNorm
5. GELUActivation
```

### heyarmar
- Layers: 255

Layer sequence (first 5):
```
1. Conv1d
2. GELUActivation
3. Conv1d
4. GELUActivation
5. GELUActivation
```