# image-segmentation Architecture Summary (arc-based)

## Overview
- Models analyzed: 24
- Total layers: 9119
- Unique layer types: 208

## Layer Type Distribution
| Layer Type | Count | Percentage |
|-----------|-------|------------|
| Linear | 2293 | 25.1% |
| LayerNorm | 954 | 10.5% |
| Dropout | 879 | 9.6% |
| Conv2d | 647 | 7.1% |
| GELUActivation | 290 | 3.2% |
| BatchNorm2d | 274 | 3.0% |
| ReLU | 260 | 2.9% |
| REBNCONV | 112 | 1.2% |
| SegformerMixFFN | 104 | 1.1% |
| SegformerSelfOutput | 104 | 1.1% |
| SegformerEfficientSelfAttention | 104 | 1.1% |
| SegformerDWConv | 104 | 1.1% |
| SegformerLayer | 104 | 1.1% |
| SegformerAttention | 104 | 1.1% |
| Sequential | 100 | 1.1% |
| SegformerDropPath | 100 | 1.1% |
| ModuleList | 74 | 0.8% |
| LeakyReLU | 66 | 0.7% |
| MobileViTV2ConvLayer | 64 | 0.7% |
| Identity | 62 | 0.7% |

## Sample Models

### mobilenet_v2_1-10k-steps
- Layers: 210

Layer sequence (first 5):
```
1. MobileNetV2Stem
2. MobileNetV2ConvLayer
3. Conv2d
4. BatchNorm2d
5. ReLU6
```

### sidewalk-semantic-demo
- Layers: 196

Layer sequence (first 5):
```
1. SegformerEncoder
2. ModuleList
3. SegformerOverlapPatchEmbeddings
4. Conv2d
5. LayerNorm
```

### FixRM
- Layers: 500

Layer sequence (first 5):
```
1. Conv2d
2. MaxPool2d
3. RSU7
4. REBNCONV
5. Conv2d
```