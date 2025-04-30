# Neural Network Architecture Patterns (arc-based)

## Task Overview
Total tasks analyzed: 7
Tasks: audio-classification, automatic-speech-recognition, feature-extraction, fill-mask, image-feature-extraction, image-segmentation, image-text-to-text

## Hierarchical Structures

### feature-extraction
| Nesting Level | Component Count | Percentage |
|--------------|----------------|------------|
| 0 | 157 | 1.2% |
| 1 | 505 | 3.9% |
| 2 | 1204 | 9.4% |
| 3 | 2379 | 18.5% |
| 4 | 3209 | 24.9% |
| 5 | 4290 | 33.3% |
| 6 | 1050 | 8.2% |
| 7 | 72 | 0.6% |

### image-text-to-text
| Nesting Level | Component Count | Percentage |
|--------------|----------------|------------|
| 0 | 27 | 0.7% |
| 1 | 40 | 1.1% |
| 2 | 59 | 1.6% |
| 3 | 252 | 6.8% |
| 4 | 817 | 22.2% |
| 5 | 1491 | 40.5% |
| 6 | 996 | 27.1% |

### audio-classification
| Nesting Level | Component Count | Percentage |
|--------------|----------------|------------|
| 0 | 39 | 0.6% |
| 1 | 134 | 1.9% |
| 2 | 419 | 5.9% |
| 3 | 2539 | 36.0% |
| 4 | 3492 | 49.5% |
| 5 | 363 | 5.1% |
| 6 | 72 | 1.0% |

### automatic-speech-recognition
| Nesting Level | Component Count | Percentage |
|--------------|----------------|------------|
| 0 | 22 | 0.5% |
| 1 | 71 | 1.7% |
| 2 | 253 | 6.0% |
| 3 | 1693 | 40.1% |
| 4 | 1946 | 46.0% |
| 5 | 242 | 5.7% |

### image-feature-extraction
| Nesting Level | Component Count | Percentage |
|--------------|----------------|------------|
| 0 | 17 | 1.6% |
| 1 | 49 | 4.5% |
| 2 | 151 | 13.9% |
| 3 | 442 | 40.6% |
| 4 | 300 | 27.5% |
| 5 | 130 | 11.9% |

### image-segmentation
| Nesting Level | Component Count | Percentage |
|--------------|----------------|------------|
| 0 | 79 | 0.9% |
| 1 | 300 | 3.3% |
| 2 | 638 | 7.0% |
| 3 | 1039 | 11.4% |
| 4 | 1526 | 16.7% |
| 5 | 2027 | 22.2% |
| 6 | 1370 | 15.0% |
| 7 | 673 | 7.4% |
| 8 | 675 | 7.4% |
| 9 | 576 | 6.3% |
| 10 | 216 | 2.4% |

### fill-mask
| Nesting Level | Component Count | Percentage |
|--------------|----------------|------------|
| 0 | 177 | 0.9% |
| 1 | 318 | 1.7% |
| 2 | 554 | 2.9% |
| 3 | 1351 | 7.1% |
| 4 | 3576 | 18.9% |
| 5 | 6916 | 36.6% |
| 6 | 5940 | 31.4% |
| 7 | 72 | 0.4% |

## Task-Specific Components

### feature-extraction
| Layer Type |
|-----------|
| BloomAttention |
| BloomBlock |
| BloomGelu |
| BloomMLP |
| CamembertPooler |
| CanineAttention |
| CanineEmbeddings |
| CanineEncoder |
| CanineIntermediate |
| CanineLayer |
| CanineOutput |
| CaninePooler |
| CanineSelfAttention |
| CanineSelfOutput |
| CharactersToMolecules |
| ConvBertAttention |
| ConvBertEmbeddings |
| ConvBertEncoder |
| ConvBertIntermediate |
| ConvBertLayer |
| ConvBertOutput |
| ConvBertSelfAttention |
| ConvBertSelfOutput |
| ConvProjection |
| DPREncoder |
| Data2VecTextAttention |
| Data2VecTextEncoder |
| Data2VecTextForTextEmbeddings |
| Data2VecTextIntermediate |
| Data2VecTextLayer |
| Data2VecTextOutput |
| Data2VecTextPooler |
| Data2VecTextSelfAttention |
| Data2VecTextSelfOutput |
| FunnelDecoder |
| FunnelEmbeddings |
| FunnelEncoder |
| FunnelLayer |
| FunnelPositionwiseFFN |
| FunnelRelMultiheadAttention |
| GPT2Attention |
| GPT2Block |
| GPT2MLP |
| GPTNeoAttention |
| GPTNeoBlock |
| GPTNeoMLP |
| GPTNeoSelfAttention |
| LongformerPooler |
| MPNetPooler |
| MT5Attention |
| MT5Block |
| MT5DenseGatedActDense |
| MT5LayerFF |
| MT5LayerNorm |
| MT5LayerSelfAttention |
| MT5Stack |
| MistralAttention |
| MistralDecoderLayer |
| MistralMLP |
| MistralRMSNorm |
| MistralRotaryEmbedding |
| MultiHeadAttention |
| PegasusAttention |
| PegasusEncoder |
| PegasusEncoderLayer |
| PegasusSinusoidalPositionalEmbedding |
| RoFormerEmbeddings |
| RobertaPooler |
| SeparableConv1D |
| T5Attention |
| T5Block |
| T5DenseActDense |
| T5DenseGatedActDense |
| T5LayerFF |
| T5LayerNorm |
| T5LayerSelfAttention |
| T5Stack |
| TapasAttention |
| TapasEmbeddings |
| TapasEncoder |
| TapasIntermediate |
| TapasLayer |
| TapasOutput |
| TapasPooler |
| TapasSelfAttention |
| TapasSelfOutput |
| TransformerFFN |
| XLMRobertaPooler |

### image-text-to-text
| Layer Type |
|-----------|
| Blip2Attention |
| Blip2Encoder |
| Blip2EncoderLayer |
| Blip2MLP |
| Blip2QFormerAttention |
| Blip2QFormerEncoder |
| Blip2QFormerIntermediate |
| Blip2QFormerLayer |
| Blip2QFormerModel |
| Blip2QFormerMultiHeadAttention |
| Blip2QFormerOutput |
| Blip2QFormerSelfOutput |
| Blip2VisionEmbeddings |
| Blip2VisionModel |
| BlipAttention |
| BlipEncoder |
| BlipEncoderLayer |
| BlipMLP |
| BlipTextAttention |
| BlipTextEmbeddings |
| BlipTextEncoder |
| BlipTextIntermediate |
| BlipTextLayer |
| BlipTextModel |
| BlipTextOutput |
| BlipTextPooler |
| BlipTextSelfAttention |
| BlipTextSelfOutput |
| BlipVisionEmbeddings |
| BlipVisionModel |
| GitAttention |
| GitEmbeddings |
| GitEncoder |
| GitIntermediate |
| GitLayer |
| GitOutput |
| GitSelfAttention |
| GitSelfOutput |
| OPTDecoder |
| OPTDecoderLayer |
| OPTForCausalLM |
| OPTLearnedPositionalEmbedding |
| OPTModel |
| OPTSdpaAttention |
| RelativePositionBias1D |

### audio-classification
| Layer Type |
|-----------|
| ASTAttention |
| ASTEmbeddings |
| ASTEncoder |
| ASTIntermediate |
| ASTLayer |
| ASTOutput |
| ASTPatchEmbeddings |
| ASTSelfAttention |
| ASTSelfOutput |
| AvgPool1d |
| BatchNorm1d |
| Data2VecAudioConvLayer |
| Data2VecAudioFeatureEncoder |
| Data2VecAudioFeedForward |
| Data2VecAudioPadLayer |
| GLU |
| HubertFeedForward |
| HubertGroupNormConvLayer |
| HubertNoLayerNormConvLayer |
| MambaBlock |
| MambaMixer |
| MambaRMSNorm |
| SEWDFeatureEncoder |
| SEWDGroupNormConvLayer |
| SEWDNoLayerNormConvLayer |
| SEWDSamePadLayer |
| StableDropout |
| UniSpeechSatLayerNormConvLayer |
| Wav2Vec2ConformerFeatureEncoder |
| Wav2Vec2ConformerFeedForward |
| Wav2Vec2ConformerLayerNormConvLayer |
| Wav2Vec2ConformerRelPositionalEmbedding |
| Wav2Vec2ConformerSamePadLayer |
| WavLMFeedForward |
| WavLMGroupNormConvLayer |
| WavLMNoLayerNormConvLayer |

### automatic-speech-recognition
| Layer Type |
|-----------|
| Conv1dSubsampler |
| MCTCTLayerNorm |
| Speech2TextAttention |
| Speech2TextDecoder |
| Speech2TextDecoderLayer |
| Speech2TextEncoder |
| Speech2TextEncoderLayer |
| Speech2TextSinusoidalPositionalEmbedding |
| SpeechT5Attention |
| SpeechT5Decoder |
| SpeechT5DecoderLayer |
| SpeechT5DecoderWithoutPrenet |
| SpeechT5Encoder |
| SpeechT5EncoderLayer |
| SpeechT5EncoderWithoutPrenet |
| SpeechT5FeedForward |
| SpeechT5RelativePositionalEncoding |

### image-feature-extraction
| Layer Type |
|-----------|
| AIMv2Attention |
| AIMv2Block |
| AIMv2PatchEmbed |
| AIMv2SwiGLUFFN |
| AIMv2Transformer |
| AIMv2ViTPreprocessor |
| Data2VecVisionAttention |
| Data2VecVisionDropPath |
| Data2VecVisionEmbeddings |
| Data2VecVisionEncoder |
| Data2VecVisionIntermediate |
| Data2VecVisionLayer |
| Data2VecVisionOutput |
| Data2VecVisionPatchEmbeddings |
| Data2VecVisionRelativePositionBias |
| Data2VecVisionSdpaSelfAttention |
| Data2VecVisionSelfOutput |
| ImageGPTAttention |
| ImageGPTBlock |
| ImageGPTLayerNorm |
| ImageGPTMLP |
| RMSNorm |
| ZeroPad2d |

### image-segmentation
| Layer Type |
|-----------|
| AdaptiveAvgPool1d |
| AdaptiveAvgPool2d |
| AvgPool2d |
| BatchNorm2d |
| BatchNorm3d |
| BeitAttention |
| BeitDropPath |
| BeitEmbeddings |
| BeitEncoder |
| BeitIntermediate |
| BeitLayer |
| BeitOutput |
| BeitPatchEmbeddings |
| BeitPooler |
| BeitRelativePositionBias |
| BeitSdpaSelfAttention |
| BeitSelfOutput |
| CLIPSegAttention |
| CLIPSegEncoder |
| CLIPSegEncoderLayer |
| CLIPSegMLP |
| CLIPSegTextEmbeddings |
| CLIPSegTextTransformer |
| CLIPSegVisionEmbeddings |
| CLIPSegVisionTransformer |
| Conv3d |
| ConvNextEmbeddings |
| ConvNextEncoder |
| ConvNextLayer |
| ConvNextLayerNorm |
| ConvNextStage |
| ConvTranspose2d |
| DPTSelfAttention |
| DPTViTAttention |
| DPTViTEmbeddings |
| DPTViTEncoder |
| DPTViTIntermediate |
| DPTViTLayer |
| DPTViTOutput |
| DPTViTPatchEmbeddings |
| DPTViTPooler |
| DPTViTSelfOutput |
| DetrAttention |
| DetrDecoder |
| DetrDecoderLayer |
| InjectionConvEncoder2D |
| InjectionUNet2D |
| InstanceNorm2d |
| LeakyReLU |
| LiltAttention |
| LiltEncoder |
| LiltIntermediate |
| LiltLayer |
| LiltLayoutEmbeddings |
| LiltOutput |
| LiltPooler |
| LiltSelfAttention |
| LiltSelfOutput |
| LiltTextEmbeddings |
| Mask2FormerAttention |
| Mask2FormerMLPPredictionHead |
| Mask2FormerMaskPredictor |
| Mask2FormerMaskedAttentionDecoder |
| Mask2FormerMaskedAttentionDecoderLayer |
| Mask2FormerPixelDecoder |
| Mask2FormerPixelDecoderEncoderLayer |
| Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention |
| Mask2FormerPixelDecoderEncoderOnly |
| Mask2FormerPixelLevelModule |
| Mask2FormerPredictionBlock |
| Mask2FormerSinePositionEmbedding |
| Mask2FormerTransformerModule |
| MaskFormerFPNConvLayer |
| MaskFormerFPNLayer |
| MaskFormerFPNModel |
| MaskFormerPixelDecoder |
| MaskFormerPixelLevelModule |
| MaskFormerSinePositionEmbedding |
| MaskFormerSwinAttention |
| MaskFormerSwinBackbone |
| MaskFormerSwinDropPath |
| MaskFormerSwinEmbeddings |
| MaskFormerSwinEncoder |
| MaskFormerSwinIntermediate |
| MaskFormerSwinLayer |
| MaskFormerSwinModel |
| MaskFormerSwinOutput |
| MaskFormerSwinPatchEmbeddings |
| MaskFormerSwinPatchMerging |
| MaskFormerSwinSelfAttention |
| MaskFormerSwinSelfOutput |
| MaskFormerSwinStage |
| MaskFormerTransformerModule |
| MaxPool2d |
| MaxPool3d |
| MobileNetV2ConvLayer |
| MobileNetV2InvertedResidual |
| MobileNetV2Stem |
| MobileViTAttention |
| MobileViTConvLayer |
| MobileViTEncoder |
| MobileViTIntermediate |
| MobileViTInvertedResidual |
| MobileViTLayer |
| MobileViTMobileNetLayer |
| MobileViTOutput |
| MobileViTSelfAttention |
| MobileViTSelfOutput |
| MobileViTTransformer |
| MobileViTTransformerLayer |
| MobileViTV2ConvLayer |
| MobileViTV2Encoder |
| MobileViTV2FFN |
| MobileViTV2InvertedResidual |
| MobileViTV2Layer |
| MobileViTV2LinearSelfAttention |
| MobileViTV2MobileNetLayer |
| MobileViTV2Transformer |
| MobileViTV2TransformerLayer |
| ModuleDict |
| MultiheadAttention |
| NonDynamicallyQuantizableLinear |
| OneFormerAttention |
| OneFormerMLPPredictionHead |
| OneFormerPixelDecoder |
| OneFormerPixelDecoderEncoderLayer |
| OneFormerPixelDecoderEncoderMultiscaleDeformableAttention |
| OneFormerPixelDecoderEncoderOnly |
| OneFormerPixelLevelModule |
| OneFormerSinePositionEmbedding |
| OneFormerTaskModel |
| OneFormerTransformerDecoder |
| OneFormerTransformerDecoderCrossAttentionLayer |
| OneFormerTransformerDecoderFFNLayer |
| OneFormerTransformerDecoderLayer |
| OneFormerTransformerDecoderQueryTransformer |
| OneFormerTransformerDecoderQueryTransformerDecoder |
| OneFormerTransformerDecoderQueryTransformerDecoderLayer |
| OneFormerTransformerDecoderSelfAttentionLayer |
| OneFormerTransformerModule |
| PredictionBlock |
| ProbabilisticSegmentationNet |
| REBNCONV |
| RSU4 |
| RSU4F |
| RSU5 |
| RSU6 |
| RSU7 |
| ReLU6 |
| ResNetBottleNeckLayer |
| ResNetConvLayer |
| ResNetEmbeddings |
| ResNetEncoder |
| ResNetShortCut |
| ResNetStage |
| SamPositionalEmbedding |
| SegformerAttention |
| SegformerDWConv |
| SegformerDropPath |
| SegformerEfficientSelfAttention |
| SegformerEncoder |
| SegformerLayer |
| SegformerMixFFN |
| SegformerOverlapPatchEmbeddings |
| SegformerSelfOutput |
| Sequential |
| Sigmoid |
| SwinAttention |
| SwinBackbone |
| SwinDropPath |
| SwinEmbeddings |
| SwinEncoder |
| SwinIntermediate |
| SwinLayer |
| SwinOutput |
| SwinPatchEmbeddings |
| SwinPatchMerging |
| SwinSelfAttention |
| SwinSelfOutput |
| SwinStage |
| U_Net |
| U_Net_DeepSup |
| Upsample |
| conv_block |
| up_conv |

### fill-mask
| Layer Type |
|-----------|
| AlbertMLMHead |
| AlbertModel |
| BertLMPredictionHead |
| BertOnlyMLMHead |
| BertPredictionHeadTransform |
| BigBirdAttention |
| BigBirdEmbeddings |
| BigBirdEncoder |
| BigBirdIntermediate |
| BigBirdLMPredictionHead |
| BigBirdLayer |
| BigBirdModel |
| BigBirdOnlyMLMHead |
| BigBirdOutput |
| BigBirdPredictionHeadTransform |
| BigBirdSelfOutput |
| CamembertLMHead |
| CamembertModel |
| DebertaV2Model |
| DistilBertModel |
| ElectraGeneratorPredictions |
| ElectraModel |
| ErnieAttention |
| ErnieEmbeddings |
| ErnieEncoder |
| ErnieIntermediate |
| ErnieLMPredictionHead |
| ErnieLayer |
| ErnieModel |
| ErnieOnlyMLMHead |
| ErnieOutput |
| ErniePredictionHeadTransform |
| ErnieSelfAttention |
| ErnieSelfOutput |
| EuroBertAttention |
| EuroBertDecoderLayer |
| EuroBertMLP |
| EuroBertModel |
| EuroBertRMSNorm |
| EuroBertRotaryEmbedding |
| LegacyDebertaV2LMPredictionHead |
| LegacyDebertaV2OnlyMLMHead |
| LegacyDebertaV2PredictionHeadTransform |
| LongformerLMHead |
| LongformerModel |
| LukeAttention |
| LukeEmbeddings |
| LukeEncoder |
| LukeIntermediate |
| LukeLMHead |
| LukeLayer |
| LukeModel |
| LukeOutput |
| LukePooler |
| LukeSelfAttention |
| LukeSelfOutput |
| MBartDecoder |
| MBartDecoderLayer |
| MBartEncoder |
| MBartEncoderLayer |
| MBartLearnedPositionalEmbedding |
| MBartModel |
| MBartScaledWordEmbedding |
| MBartSdpaAttention |
| MPNetLMHead |
| MPNetModel |
| MegatronBertAttention |
| MegatronBertEmbeddings |
| MegatronBertEncoder |
| MegatronBertIntermediate |
| MegatronBertLMPredictionHead |
| MegatronBertLayer |
| MegatronBertModel |
| MegatronBertOnlyMLMHead |
| MegatronBertOutput |
| MegatronBertPredictionHeadTransform |
| MegatronBertSelfAttention |
| MegatronBertSelfOutput |
| ModernBertModel |
| ModernBertPredictionHead |
| NezhaAttention |
| NezhaEmbeddings |
| NezhaEncoder |
| NezhaIntermediate |
| NezhaLMPredictionHead |
| NezhaLayer |
| NezhaModel |
| NezhaOnlyMLMHead |
| NezhaOutput |
| NezhaPredictionHeadTransform |
| NezhaRelativePositionsEncoding |
| NezhaSelfAttention |
| NezhaSelfOutput |
| RobertaLMHead |
| RobertaModel |
| XLMRobertaLMHead |
| XLMRobertaModel |