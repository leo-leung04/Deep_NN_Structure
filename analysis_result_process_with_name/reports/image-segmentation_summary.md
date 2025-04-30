# image-segmentation Architecture Summary (name-based)

## Overview
- Models analyzed: 61
- Total layers: 32157
- Unique layer types: 199

## Layer Type Distribution
| Layer Type | Count | Percentage |
|-----------|-------|------------|
| Linear | 7364 | 22.9% |
| LayerNorm | 3574 | 11.1% |
| Dropout | 3364 | 10.5% |
| Conv2d | 3156 | 9.8% |
| BatchNorm2d | 1147 | 3.6% |
| GELUActivation | 1126 | 3.5% |
| ReLU | 1086 | 3.4% |
| SegformerAttention | 916 | 2.8% |
| SegformerMixFFN | 916 | 2.8% |
| SegformerSelfOutput | 916 | 2.8% |
| SegformerEfficientSelfAttention | 916 | 2.8% |
| SegformerDWConv | 916 | 2.8% |
| SegformerLayer | 916 | 2.8% |
| SegformerDropPath | 890 | 2.8% |
| ResNetConvLayer | 539 | 1.7% |
| REBNCONV | 448 | 1.4% |
| Identity | 364 | 1.1% |
| Sequential | 300 | 0.9% |
| ModuleList | 239 | 0.7% |
| ResNetBottleNeckLayer | 176 | 0.5% |

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

### dropoff-utcustom-train-SF-RGB-b0_1
- Layers: 196

Layer sequence (first 5):
```
1. SegformerEncoder
2. ModuleList
3. SegformerOverlapPatchEmbeddings
4. Conv2d
5. LayerNorm
```