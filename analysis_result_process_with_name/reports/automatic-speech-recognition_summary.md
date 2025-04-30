# automatic-speech-recognition Architecture Summary (name-based)

## Overview
- Models analyzed: 137
- Total layers: 28734
- Unique layer types: 58

## Layer Type Distribution
| Layer Type | Count | Percentage |
|-----------|-------|------------|
| Dropout | 9005 | 31.3% |
| Linear | 6076 | 21.1% |
| GELUActivation | 5380 | 18.7% |
| LayerNorm | 2600 | 9.0% |
| Conv1d | 1135 | 4.0% |
| _WeightNorm | 875 | 3.0% |
| ParametrizationList | 875 | 3.0% |
| Wav2Vec2LayerNormConvLayer | 704 | 2.5% |
| WhisperSdpaAttention | 552 | 1.9% |
| Wav2Vec2FeedForward | 384 | 1.3% |
| Wav2Vec2NoLayerNormConvLayer | 198 | 0.7% |
| WhisperDecoderLayer | 184 | 0.6% |
| WhisperEncoderLayer | 184 | 0.6% |
| Wav2Vec2FeatureEncoder | 122 | 0.4% |
| Wav2Vec2SamePadLayer | 122 | 0.4% |
| GroupNorm | 34 | 0.1% |
| Wav2Vec2GroupNormConvLayer | 33 | 0.1% |
| Speech2TextAttention | 24 | 0.1% |
| SpeechT5Attention | 24 | 0.1% |
| ModuleList | 21 | 0.1% |

## Sample Models

### wav2vec2test
- Layers: 255

Layer sequence (first 5):
```
1. Conv1d
2. GELUActivation
3. Conv1d
4. GELUActivation
5. GELUActivation
```

### s2t-large-librispeech-asr
- Layers: 247

Layer sequence (first 5):
```
1. Speech2TextEncoder
2. Conv1dSubsampler
3. ModuleList
4. Conv1d
5. Conv1d
```

### sat-base
- Layers: 255

Layer sequence (first 5):
```
1. Conv1d
2. GELUActivation
3. Conv1d
4. GELUActivation
5. GELUActivation
```