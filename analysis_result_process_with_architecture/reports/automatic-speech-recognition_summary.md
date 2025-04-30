# automatic-speech-recognition Architecture Summary (arc-based)

## Overview
- Models analyzed: 16
- Total layers: 4227
- Unique layer types: 58

## Layer Type Distribution
| Layer Type | Count | Percentage |
|-----------|-------|------------|
| Linear | 1680 | 39.7% |
| Dropout | 635 | 15.0% |
| LayerNorm | 573 | 13.6% |
| GELUActivation | 487 | 11.5% |
| WhisperSdpaAttention | 216 | 5.1% |
| Conv1d | 78 | 1.8% |
| WhisperEncoderLayer | 72 | 1.7% |
| WhisperDecoderLayer | 72 | 1.7% |
| _WeightNorm | 56 | 1.3% |
| ParametrizationList | 56 | 1.3% |
| SpeechT5Attention | 24 | 0.6% |
| Speech2TextAttention | 24 | 0.6% |
| Wav2Vec2LayerNormConvLayer | 20 | 0.5% |
| ReLU | 18 | 0.4% |
| SpeechT5FeedForward | 18 | 0.4% |
| BartSdpaAttention | 18 | 0.4% |
| ModuleList | 13 | 0.3% |
| Wav2Vec2FeedForward | 12 | 0.3% |
| Wav2Vec2NoLayerNormConvLayer | 12 | 0.3% |
| SpeechT5EncoderLayer | 12 | 0.3% |

## Sample Models

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

### repo
- Layers: 155

Layer sequence (first 5):
```
1. Conv1d
2. GELUActivation
3. Conv1d
4. LayerNorm
5. GELUActivation
```