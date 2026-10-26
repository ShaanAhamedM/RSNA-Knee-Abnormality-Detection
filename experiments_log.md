# RSNA Knee Abnormality Detection — Autonomous Tournament Log

**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6.0 GB VRAM)  
**Evaluation Metric:** Macro-Averaged ROC-AUC (12 Knee Pathology Classes)  
**Data Strategy:** Leakage-Free 5-Fold Multilabel Stratified Split (`folds.csv`) + Zero-Unzip In-Memory Streaming  
**Target Schema:** `['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']`  

---

## Tournament Progress & Leaderboard

| Iteration | Model Family | Backbone | Loss Formulation | Res | Slices (K) | 5-Fold OOF Macro-AUC | Ensemble OOF Macro-AUC | Submission Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | Prior Baseline | Uniform Class Prior | N/A | N/A | N/A | 0.5000 | 0.5000 | Generated |

---

| 1 | ConvNet | efficientnet_b0 | poly(eps=1.0) | 256x256 | 6 | 0.4779 | 0.4856 | Updated |

| 2 | ConvNeXt | convnext_small | poly(eps=1.0) | 256x256 | 6 | 0.5455 | 0.4988 | Retained |

| 3 | ResNet/SENet | seresnext50_32x4d | asl | 256x256 | 6 | 0.5592 | 0.5007 | Retained |

| 4 | Vision Transformer | swin_tiny_patch4_window7_224 | poly(eps=0.5) | 224x224 | 6 | 0.5222 | 0.5063 | Retained |

| 5 | DenseNet | densenet121 | uncertainty | 256x256 | 8 | 0.5358 | 0.5058 | Retained |

| 6 | Volumetric ConvNeXt | convnext_small | poly(eps=1.5) | 256x256 | 10 | 0.5575 | 0.5176 | Retained |

| 7 | EfficientNet | efficientnet_b2 | poly(eps=1.0) | 288x288 | 6 | 0.5338 | 0.5147 | Retained |

| 8 | Dynamic-Perturbation-1 | seresnext50_32x4d | poly(eps=2.0) | 256x256 | 8 | 0.5683 | 0.5206 | Retained |

| 9 | Dynamic-Perturbation-2 | efficientnet_b2 | poly(eps=0.2) | 256x256 | 6 | 0.5274 | 0.5239 | Retained |

| 10 | Dynamic-Perturbation-3 | densenet121 | poly(eps=1.2) | 256x256 | 6 | 0.5369 | 0.5181 | Retained |

| 11 | Dynamic-Perturbation-4 | convnext_small | poly(eps=1.8) | 256x256 | 6 | 0.5587 | 0.5312 | Retained |

| 12 | Dynamic-Perturbation-5 | seresnext50_32x4d | poly(eps=0.5) | 256x256 | 8 | 0.4882 | 0.5235 | Updated |

| 13 | Dynamic-Perturbation-6 | efficientnet_b2 | poly(eps=0.5) | 256x256 | 8 | 0.5199 | 0.5267 | Updated |

| 14 | Dynamic-Perturbation-7 | densenet121 | poly(eps=0.5) | 256x256 | 4 | 0.5047 | 0.5137 | Updated |

| 15 | Dynamic-Perturbation-8 | convnext_small | poly(eps=0.2) | 256x256 | 8 | 0.5893 | 0.5104 | Retained |

| 16 | Dynamic-Perturbation-9 | seresnext50_32x4d | poly(eps=1.2) | 256x256 | 8 | 0.4850 | 0.5108 | Updated |

| 17 | Dynamic-Perturbation-10 | efficientnet_b2 | poly(eps=1.2) | 256x256 | 6 | 0.4871 | 0.5028 | Updated |

| 18 | Dynamic-Perturbation-11 | densenet121 | poly(eps=1.2) | 256x256 | 8 | 0.4941 | 0.5023 | Updated |

| 19 | Dynamic-Perturbation-12 | convnext_small | poly(eps=1.8) | 256x256 | 8 | 0.5758 | 0.5035 | Retained |

| 20 | Dynamic-Perturbation-13 | seresnext50_32x4d | poly(eps=0.2) | 256x256 | 6 | 0.5101 | 0.5093 | Retained |

| 21 | Dynamic-Perturbation-14 | efficientnet_b2 | poly(eps=0.5) | 256x256 | 8 | 0.4975 | 0.5092 | Updated |

| 22 | Dynamic-Perturbation-15 | densenet121 | poly(eps=1.8) | 256x256 | 6 | 0.5141 | 0.5147 | Updated |

| 23 | Dynamic-Perturbation-16 | convnext_small | poly(eps=0.5) | 256x256 | 8 | 0.5796 | 0.4983 | Retained |

| 24 | Dynamic-Perturbation-17 | seresnext50_32x4d | poly(eps=2.0) | 256x256 | 8 | 0.5340 | 0.5066 | Retained |

| 25 | Dynamic-Perturbation-18 | efficientnet_b2 | poly(eps=0.5) | 256x256 | 6 | 0.5288 | 0.5117 | Retained |

| 26 | Dynamic-Perturbation-19 | densenet121 | poly(eps=1.2) | 256x256 | 10 | 0.5155 | 0.5016 | Retained |

| 27 | Dynamic-Perturbation-20 | convnext_small | poly(eps=0.5) | 256x256 | 10 | 0.5860 | 0.5135 | Retained |

| 28 | Dynamic-Perturbation-21 | seresnext50_32x4d | poly(eps=1.2) | 256x256 | 10 | 0.5157 | 0.5078 | Retained |

| 29 | Dynamic-Perturbation-22 | efficientnet_b2 | poly(eps=0.2) | 256x256 | 8 | 0.4851 | 0.5003 | Updated |

| 30 | Dynamic-Perturbation-23 | densenet121 | poly(eps=0.2) | 256x256 | 6 | 0.5381 | 0.5092 | Retained |

| 31 | Dynamic-Perturbation-24 | convnext_small | poly(eps=1.2) | 256x256 | 10 | 0.5705 | 0.5153 | Retained |

| 32 | Dynamic-Perturbation-25 | seresnext50_32x4d | poly(eps=0.2) | 256x256 | 4 | 0.5104 | 0.5171 | Updated |

| 33 | Dynamic-Perturbation-26 | efficientnet_b2 | poly(eps=0.5) | 256x256 | 10 | 0.5148 | 0.5156 | Updated |

| 34 | Dynamic-Perturbation-27 | densenet121 | poly(eps=1.8) | 256x256 | 6 | 0.4921 | 0.4964 | Updated |

| 35 | Dynamic-Perturbation-28 | convnext_small | poly(eps=0.2) | 256x256 | 10 | 0.5903 | 0.5084 | Retained |

| 36 | Dynamic-Perturbation-29 | seresnext50_32x4d | poly(eps=1.2) | 256x256 | 10 | 0.5891 | 0.5153 | Retained |

| 37 | Dynamic-Perturbation-30 | efficientnet_b2 | poly(eps=0.2) | 256x256 | 10 | 0.5378 | 0.5183 | Retained |

| 38 | Dynamic-Perturbation-31 | densenet121 | poly(eps=0.2) | 256x256 | 4 | 0.4920 | 0.5232 | Updated |

| 39 | Dynamic-Perturbation-32 | convnext_small | poly(eps=0.2) | 256x256 | 8 | 0.5835 | 0.5181 | Retained |

| 40 | Dynamic-Perturbation-33 | seresnext50_32x4d | poly(eps=1.8) | 256x256 | 4 | 0.5071 | 0.5110 | Updated |

| 41 | Dynamic-Perturbation-34 | efficientnet_b2 | poly(eps=1.2) | 256x256 | 8 | 0.5055 | 0.5090 | Updated |

| 42 | Dynamic-Perturbation-35 | densenet121 | poly(eps=0.2) | 256x256 | 6 | 0.5567 | 0.5276 | Retained |

| 43 | Dynamic-Perturbation-36 | convnext_small | poly(eps=1.8) | 256x256 | 4 | 0.5710 | 0.5325 | Retained |

| 44 | Dynamic-Perturbation-37 | seresnext50_32x4d | poly(eps=0.2) | 256x256 | 6 | 0.5275 | 0.5314 | Updated |

| 45 | Dynamic-Perturbation-38 | efficientnet_b2 | poly(eps=0.2) | 256x256 | 8 | 0.5497 | 0.5313 | Retained |

| 46 | Dynamic-Perturbation-39 | densenet121 | poly(eps=0.5) | 256x256 | 4 | 0.4863 | 0.5098 | Updated |

| 47 | Dynamic-Perturbation-40 | convnext_small | poly(eps=1.8) | 256x256 | 8 | 0.5715 | 0.5010 | Retained |

| 48 | Dynamic-Perturbation-41 | seresnext50_32x4d | poly(eps=0.5) | 256x256 | 4 | 0.5658 | 0.5059 | Retained |

| 49 | Dynamic-Perturbation-42 | efficientnet_b2 | poly(eps=2.0) | 256x256 | 6 | 0.4926 | 0.5001 | Updated |

| 50 | Dynamic-Perturbation-43 | densenet121 | poly(eps=2.0) | 256x256 | 4 | 0.4742 | 0.5056 | Updated |

| 51 | Dynamic-Perturbation-44 | convnext_small | poly(eps=0.2) | 256x256 | 6 | 0.5874 | 0.5023 | Retained |

| 52 | Dynamic-Perturbation-45 | seresnext50_32x4d | poly(eps=2.0) | 256x256 | 6 | 0.5348 | 0.5083 | Retained |

| 53 | Dynamic-Perturbation-46 | efficientnet_b2 | poly(eps=0.5) | 256x256 | 8 | 0.5372 | 0.5127 | Retained |

| 54 | Dynamic-Perturbation-47 | densenet121 | poly(eps=1.2) | 256x256 | 4 | 0.4715 | 0.5107 | Updated |

| 55 | Dynamic-Perturbation-48 | convnext_small | poly(eps=1.2) | 256x256 | 8 | 0.5470 | 0.5035 | Retained |

## Detailed Iteration Logs

### Iteration 0: Pipeline Lock-In & Environment Verification
- **Status:** Complete
- **Zero-Unzip Streaming:** Verified direct in-memory DICOM parsing from `D:\RSNA\rsna-knee-abnormality-detection.zip` (819,635 slice Central Directory indexed).
- **VRAM Utilization:** 967 MB allocated / 1,440 MB reserved (< 25% of 6 GB ceiling).
- **Initial Submission:** Verified matching schema, zero NaNs, bounds in [0.001, 0.999].

### Iteration 1: ConvNet (efficientnet_b0)
- **Loss Formulation:** poly(eps=1.0)
- **Configuration:** Resolution=256x256, Slices=6
- **5-Fold OOF Macro-AUC:** 0.4779
- **Ensemble OOF Macro-AUC:** 0.4856
- **Notes:** Baseline EfficientNet-B0 with 2.5D Gated Attention-MIL and PolyLoss (eps=1.0)

### Iteration 2: ConvNeXt (convnext_small)
- **Loss Formulation:** poly(eps=1.0)
- **Configuration:** Resolution=256x256, Slices=6
- **5-Fold OOF Macro-AUC:** 0.5455
- **Ensemble OOF Macro-AUC:** 0.4988
- **Notes:** ConvNeXt-Small with Gated Attention-MIL and Gradient Checkpointing

### Iteration 3: ResNet/SENet (seresnext50_32x4d)
- **Loss Formulation:** asl
- **Configuration:** Resolution=256x256, Slices=6
- **5-Fold OOF Macro-AUC:** 0.5592
- **Ensemble OOF Macro-AUC:** 0.5007
- **Notes:** SE-ResNeXt50 with Asymmetric Focal Loss for extreme class imbalance

### Iteration 4: Vision Transformer (swin_tiny_patch4_window7_224)
- **Loss Formulation:** poly(eps=0.5)
- **Configuration:** Resolution=224x224, Slices=6
- **5-Fold OOF Macro-AUC:** 0.5222
- **Ensemble OOF Macro-AUC:** 0.5063
- **Notes:** Swin-Tiny Transformer with shifted window attention across knee joint planes

### Iteration 5: DenseNet (densenet121)
- **Loss Formulation:** uncertainty
- **Configuration:** Resolution=256x256, Slices=8
- **5-Fold OOF Macro-AUC:** 0.5358
- **Ensemble OOF Macro-AUC:** 0.5058
- **Notes:** DenseNet-121 with 8-slice volumetric depth and Kendall homoscedastic uncertainty loss

### Iteration 6: Volumetric ConvNeXt (convnext_small)
- **Loss Formulation:** poly(eps=1.5)
- **Configuration:** Resolution=256x256, Slices=10
- **5-Fold OOF Macro-AUC:** 0.5575
- **Ensemble OOF Macro-AUC:** 0.5176
- **Notes:** ConvNeXt-Small with expanded K=10 slice bag covering 90% of joint volume

### Iteration 7: EfficientNet (efficientnet_b2)
- **Loss Formulation:** poly(eps=1.0)
- **Configuration:** Resolution=288x288, Slices=6
- **5-Fold OOF Macro-AUC:** 0.5338
- **Ensemble OOF Macro-AUC:** 0.5147
- **Notes:** High-resolution 288x288 EfficientNet-B2 focusing on delicate cartilage and meniscus fissures

### Iteration 8: Dynamic-Perturbation-1 (seresnext50_32x4d)
- **Loss Formulation:** poly(eps=2.0)
- **Configuration:** Resolution=256x256, Slices=8
- **5-Fold OOF Macro-AUC:** 0.5683
- **Ensemble OOF Macro-AUC:** 0.5206
- **Notes:** Auto-Perturbation: eps=2.0, slices=8

### Iteration 9: Dynamic-Perturbation-2 (efficientnet_b2)
- **Loss Formulation:** poly(eps=0.2)
- **Configuration:** Resolution=256x256, Slices=6
- **5-Fold OOF Macro-AUC:** 0.5274
- **Ensemble OOF Macro-AUC:** 0.5239
- **Notes:** Auto-Perturbation: eps=0.2, slices=6

### Iteration 10: Dynamic-Perturbation-3 (densenet121)
- **Loss Formulation:** poly(eps=1.2)
- **Configuration:** Resolution=256x256, Slices=6
- **5-Fold OOF Macro-AUC:** 0.5369
- **Ensemble OOF Macro-AUC:** 0.5181
- **Notes:** Auto-Perturbation: eps=1.2, slices=6

### Iteration 11: Dynamic-Perturbation-4 (convnext_small)
- **Loss Formulation:** poly(eps=1.8)
- **Configuration:** Resolution=256x256, Slices=6
- **5-Fold OOF Macro-AUC:** 0.5587
- **Ensemble OOF Macro-AUC:** 0.5312
- **Notes:** Auto-Perturbation: eps=1.8, slices=6

### Iteration 12: Dynamic-Perturbation-5 (seresnext50_32x4d)
- **Loss Formulation:** poly(eps=0.5)
- **Configuration:** Resolution=256x256, Slices=8
- **5-Fold OOF Macro-AUC:** 0.4882
- **Ensemble OOF Macro-AUC:** 0.5235
- **Notes:** Auto-Perturbation: eps=0.5, slices=8

### Iteration 13: Dynamic-Perturbation-6 (efficientnet_b2)
- **Loss Formulation:** poly(eps=0.5)
- **Configuration:** Resolution=256x256, Slices=8
- **5-Fold OOF Macro-AUC:** 0.5199
- **Ensemble OOF Macro-AUC:** 0.5267
- **Notes:** Auto-Perturbation: eps=0.5, slices=8

### Iteration 14: Dynamic-Perturbation-7 (densenet121)
- **Loss Formulation:** poly(eps=0.5)
- **Configuration:** Resolution=256x256, Slices=4
- **5-Fold OOF Macro-AUC:** 0.5047
- **Ensemble OOF Macro-AUC:** 0.5137
- **Notes:** Auto-Perturbation: eps=0.5, slices=4

### Iteration 15: Dynamic-Perturbation-8 (convnext_small)
- **Loss Formulation:** poly(eps=0.2)
- **Configuration:** Resolution=256x256, Slices=8
- **5-Fold OOF Macro-AUC:** 0.5893
- **Ensemble OOF Macro-AUC:** 0.5104
- **Notes:** Auto-Perturbation: eps=0.2, slices=8

### Iteration 16: Dynamic-Perturbation-9 (seresnext50_32x4d)
- **Loss Formulation:** poly(eps=1.2)
- **Configuration:** Resolution=256x256, Slices=8
- **5-Fold OOF Macro-AUC:** 0.4850
- **Ensemble OOF Macro-AUC:** 0.5108
- **Notes:** Auto-Perturbation: eps=1.2, slices=8

### Iteration 17: Dynamic-Perturbation-10 (efficientnet_b2)
- **Loss Formulation:** poly(eps=1.2)
- **Configuration:** Resolution=256x256, Slices=6
- **5-Fold OOF Macro-AUC:** 0.4871
- **Ensemble OOF Macro-AUC:** 0.5028
- **Notes:** Auto-Perturbation: eps=1.2, slices=6

### Iteration 18: Dynamic-Perturbation-11 (densenet121)
- **Loss Formulation:** poly(eps=1.2)
- **Configuration:** Resolution=256x256, Slices=8
- **5-Fold OOF Macro-AUC:** 0.4941
- **Ensemble OOF Macro-AUC:** 0.5023
- **Notes:** Auto-Perturbation: eps=1.2, slices=8

### Iteration 19: Dynamic-Perturbation-12 (convnext_small)
- **Loss Formulation:** poly(eps=1.8)
- **Configuration:** Resolution=256x256, Slices=8
- **5-Fold OOF Macro-AUC:** 0.5758
- **Ensemble OOF Macro-AUC:** 0.5035
- **Notes:** Auto-Perturbation: eps=1.8, slices=8

### Iteration 20: Dynamic-Perturbation-13 (seresnext50_32x4d)
- **Loss Formulation:** poly(eps=0.2)
- **Configuration:** Resolution=256x256, Slices=6
- **5-Fold OOF Macro-AUC:** 0.5101
- **Ensemble OOF Macro-AUC:** 0.5093
- **Notes:** Auto-Perturbation: eps=0.2, slices=6

### Iteration 21: Dynamic-Perturbation-14 (efficientnet_b2)
- **Loss Formulation:** poly(eps=0.5)
- **Configuration:** Resolution=256x256, Slices=8
- **5-Fold OOF Macro-AUC:** 0.4975
- **Ensemble OOF Macro-AUC:** 0.5092
- **Notes:** Auto-Perturbation: eps=0.5, slices=8

### Iteration 22: Dynamic-Perturbation-15 (densenet121)
- **Loss Formulation:** poly(eps=1.8)
- **Configuration:** Resolution=256x256, Slices=6
- **5-Fold OOF Macro-AUC:** 0.5141
- **Ensemble OOF Macro-AUC:** 0.5147
- **Notes:** Auto-Perturbation: eps=1.8, slices=6

### Iteration 23: Dynamic-Perturbation-16 (convnext_small)
- **Loss Formulation:** poly(eps=0.5)
- **Configuration:** Resolution=256x256, Slices=8
- **5-Fold OOF Macro-AUC:** 0.5796
- **Ensemble OOF Macro-AUC:** 0.4983
- **Notes:** Auto-Perturbation: eps=0.5, slices=8

### Iteration 24: Dynamic-Perturbation-17 (seresnext50_32x4d)
- **Loss Formulation:** poly(eps=2.0)
- **Configuration:** Resolution=256x256, Slices=8
- **5-Fold OOF Macro-AUC:** 0.5340
- **Ensemble OOF Macro-AUC:** 0.5066
- **Notes:** Auto-Perturbation: eps=2.0, slices=8

### Iteration 25: Dynamic-Perturbation-18 (efficientnet_b2)
- **Loss Formulation:** poly(eps=0.5)
- **Configuration:** Resolution=256x256, Slices=6
- **5-Fold OOF Macro-AUC:** 0.5288
- **Ensemble OOF Macro-AUC:** 0.5117
- **Notes:** Auto-Perturbation: eps=0.5, slices=6

### Iteration 26: Dynamic-Perturbation-19 (densenet121)
- **Loss Formulation:** poly(eps=1.2)
- **Configuration:** Resolution=256x256, Slices=10
- **5-Fold OOF Macro-AUC:** 0.5155
- **Ensemble OOF Macro-AUC:** 0.5016
- **Notes:** Auto-Perturbation: eps=1.2, slices=10

### Iteration 27: Dynamic-Perturbation-20 (convnext_small)
- **Loss Formulation:** poly(eps=0.5)
- **Configuration:** Resolution=256x256, Slices=10
- **5-Fold OOF Macro-AUC:** 0.5860
- **Ensemble OOF Macro-AUC:** 0.5135
- **Notes:** Auto-Perturbation: eps=0.5, slices=10

### Iteration 28: Dynamic-Perturbation-21 (seresnext50_32x4d)
- **Loss Formulation:** poly(eps=1.2)
- **Configuration:** Resolution=256x256, Slices=10
- **5-Fold OOF Macro-AUC:** 0.5157
- **Ensemble OOF Macro-AUC:** 0.5078
- **Notes:** Auto-Perturbation: eps=1.2, slices=10

### Iteration 29: Dynamic-Perturbation-22 (efficientnet_b2)
- **Loss Formulation:** poly(eps=0.2)
- **Configuration:** Resolution=256x256, Slices=8
- **5-Fold OOF Macro-AUC:** 0.4851
- **Ensemble OOF Macro-AUC:** 0.5003
- **Notes:** Auto-Perturbation: eps=0.2, slices=8

### Iteration 30: Dynamic-Perturbation-23 (densenet121)
- **Loss Formulation:** poly(eps=0.2)
- **Configuration:** Resolution=256x256, Slices=6
- **5-Fold OOF Macro-AUC:** 0.5381
- **Ensemble OOF Macro-AUC:** 0.5092
- **Notes:** Auto-Perturbation: eps=0.2, slices=6

### Iteration 31: Dynamic-Perturbation-24 (convnext_small)
- **Loss Formulation:** poly(eps=1.2)
- **Configuration:** Resolution=256x256, Slices=10
- **5-Fold OOF Macro-AUC:** 0.5705
- **Ensemble OOF Macro-AUC:** 0.5153
- **Notes:** Auto-Perturbation: eps=1.2, slices=10

### Iteration 32: Dynamic-Perturbation-25 (seresnext50_32x4d)
- **Loss Formulation:** poly(eps=0.2)
- **Configuration:** Resolution=256x256, Slices=4
- **5-Fold OOF Macro-AUC:** 0.5104
- **Ensemble OOF Macro-AUC:** 0.5171
- **Notes:** Auto-Perturbation: eps=0.2, slices=4

### Iteration 33: Dynamic-Perturbation-26 (efficientnet_b2)
- **Loss Formulation:** poly(eps=0.5)
- **Configuration:** Resolution=256x256, Slices=10
- **5-Fold OOF Macro-AUC:** 0.5148
- **Ensemble OOF Macro-AUC:** 0.5156
- **Notes:** Auto-Perturbation: eps=0.5, slices=10

### Iteration 34: Dynamic-Perturbation-27 (densenet121)
- **Loss Formulation:** poly(eps=1.8)
- **Configuration:** Resolution=256x256, Slices=6
- **5-Fold OOF Macro-AUC:** 0.4921
- **Ensemble OOF Macro-AUC:** 0.4964
- **Notes:** Auto-Perturbation: eps=1.8, slices=6

### Iteration 35: Dynamic-Perturbation-28 (convnext_small)
- **Loss Formulation:** poly(eps=0.2)
- **Configuration:** Resolution=256x256, Slices=10
- **5-Fold OOF Macro-AUC:** 0.5903
- **Ensemble OOF Macro-AUC:** 0.5084
- **Notes:** Auto-Perturbation: eps=0.2, slices=10

### Iteration 36: Dynamic-Perturbation-29 (seresnext50_32x4d)
- **Loss Formulation:** poly(eps=1.2)
- **Configuration:** Resolution=256x256, Slices=10
- **5-Fold OOF Macro-AUC:** 0.5891
- **Ensemble OOF Macro-AUC:** 0.5153
- **Notes:** Auto-Perturbation: eps=1.2, slices=10

### Iteration 37: Dynamic-Perturbation-30 (efficientnet_b2)
- **Loss Formulation:** poly(eps=0.2)
- **Configuration:** Resolution=256x256, Slices=10
- **5-Fold OOF Macro-AUC:** 0.5378
- **Ensemble OOF Macro-AUC:** 0.5183
- **Notes:** Auto-Perturbation: eps=0.2, slices=10

### Iteration 38: Dynamic-Perturbation-31 (densenet121)
- **Loss Formulation:** poly(eps=0.2)
- **Configuration:** Resolution=256x256, Slices=4
- **5-Fold OOF Macro-AUC:** 0.4920
- **Ensemble OOF Macro-AUC:** 0.5232
- **Notes:** Auto-Perturbation: eps=0.2, slices=4

### Iteration 39: Dynamic-Perturbation-32 (convnext_small)
- **Loss Formulation:** poly(eps=0.2)
- **Configuration:** Resolution=256x256, Slices=8
- **5-Fold OOF Macro-AUC:** 0.5835
- **Ensemble OOF Macro-AUC:** 0.5181
- **Notes:** Auto-Perturbation: eps=0.2, slices=8

### Iteration 40: Dynamic-Perturbation-33 (seresnext50_32x4d)
- **Loss Formulation:** poly(eps=1.8)
- **Configuration:** Resolution=256x256, Slices=4
- **5-Fold OOF Macro-AUC:** 0.5071
- **Ensemble OOF Macro-AUC:** 0.5110
- **Notes:** Auto-Perturbation: eps=1.8, slices=4

### Iteration 41: Dynamic-Perturbation-34 (efficientnet_b2)
- **Loss Formulation:** poly(eps=1.2)
- **Configuration:** Resolution=256x256, Slices=8
- **5-Fold OOF Macro-AUC:** 0.5055
- **Ensemble OOF Macro-AUC:** 0.5090
- **Notes:** Auto-Perturbation: eps=1.2, slices=8

### Iteration 42: Dynamic-Perturbation-35 (densenet121)
- **Loss Formulation:** poly(eps=0.2)
- **Configuration:** Resolution=256x256, Slices=6
- **5-Fold OOF Macro-AUC:** 0.5567
- **Ensemble OOF Macro-AUC:** 0.5276
- **Notes:** Auto-Perturbation: eps=0.2, slices=6

### Iteration 43: Dynamic-Perturbation-36 (convnext_small)
- **Loss Formulation:** poly(eps=1.8)
- **Configuration:** Resolution=256x256, Slices=4
- **5-Fold OOF Macro-AUC:** 0.5710
- **Ensemble OOF Macro-AUC:** 0.5325
- **Notes:** Auto-Perturbation: eps=1.8, slices=4

### Iteration 44: Dynamic-Perturbation-37 (seresnext50_32x4d)
- **Loss Formulation:** poly(eps=0.2)
- **Configuration:** Resolution=256x256, Slices=6
- **5-Fold OOF Macro-AUC:** 0.5275
- **Ensemble OOF Macro-AUC:** 0.5314
- **Notes:** Auto-Perturbation: eps=0.2, slices=6

### Iteration 45: Dynamic-Perturbation-38 (efficientnet_b2)
- **Loss Formulation:** poly(eps=0.2)
- **Configuration:** Resolution=256x256, Slices=8
- **5-Fold OOF Macro-AUC:** 0.5497
- **Ensemble OOF Macro-AUC:** 0.5313
- **Notes:** Auto-Perturbation: eps=0.2, slices=8

### Iteration 46: Dynamic-Perturbation-39 (densenet121)
- **Loss Formulation:** poly(eps=0.5)
- **Configuration:** Resolution=256x256, Slices=4
- **5-Fold OOF Macro-AUC:** 0.4863
- **Ensemble OOF Macro-AUC:** 0.5098
- **Notes:** Auto-Perturbation: eps=0.5, slices=4

### Iteration 47: Dynamic-Perturbation-40 (convnext_small)
- **Loss Formulation:** poly(eps=1.8)
- **Configuration:** Resolution=256x256, Slices=8
- **5-Fold OOF Macro-AUC:** 0.5715
- **Ensemble OOF Macro-AUC:** 0.5010
- **Notes:** Auto-Perturbation: eps=1.8, slices=8

### Iteration 48: Dynamic-Perturbation-41 (seresnext50_32x4d)
- **Loss Formulation:** poly(eps=0.5)
- **Configuration:** Resolution=256x256, Slices=4
- **5-Fold OOF Macro-AUC:** 0.5658
- **Ensemble OOF Macro-AUC:** 0.5059
- **Notes:** Auto-Perturbation: eps=0.5, slices=4

### Iteration 49: Dynamic-Perturbation-42 (efficientnet_b2)
- **Loss Formulation:** poly(eps=2.0)
- **Configuration:** Resolution=256x256, Slices=6
- **5-Fold OOF Macro-AUC:** 0.4926
- **Ensemble OOF Macro-AUC:** 0.5001
- **Notes:** Auto-Perturbation: eps=2.0, slices=6

### Iteration 50: Dynamic-Perturbation-43 (densenet121)
- **Loss Formulation:** poly(eps=2.0)
- **Configuration:** Resolution=256x256, Slices=4
- **5-Fold OOF Macro-AUC:** 0.4742
- **Ensemble OOF Macro-AUC:** 0.5056
- **Notes:** Auto-Perturbation: eps=2.0, slices=4

### Iteration 51: Dynamic-Perturbation-44 (convnext_small)
- **Loss Formulation:** poly(eps=0.2)
- **Configuration:** Resolution=256x256, Slices=6
- **5-Fold OOF Macro-AUC:** 0.5874
- **Ensemble OOF Macro-AUC:** 0.5023
- **Notes:** Auto-Perturbation: eps=0.2, slices=6

### Iteration 52: Dynamic-Perturbation-45 (seresnext50_32x4d)
- **Loss Formulation:** poly(eps=2.0)
- **Configuration:** Resolution=256x256, Slices=6
- **5-Fold OOF Macro-AUC:** 0.5348
- **Ensemble OOF Macro-AUC:** 0.5083
- **Notes:** Auto-Perturbation: eps=2.0, slices=6

### Iteration 53: Dynamic-Perturbation-46 (efficientnet_b2)
- **Loss Formulation:** poly(eps=0.5)
- **Configuration:** Resolution=256x256, Slices=8
- **5-Fold OOF Macro-AUC:** 0.5372
- **Ensemble OOF Macro-AUC:** 0.5127
- **Notes:** Auto-Perturbation: eps=0.5, slices=8

### Iteration 54: Dynamic-Perturbation-47 (densenet121)
- **Loss Formulation:** poly(eps=1.2)
- **Configuration:** Resolution=256x256, Slices=4
- **5-Fold OOF Macro-AUC:** 0.4715
- **Ensemble OOF Macro-AUC:** 0.5107
- **Notes:** Auto-Perturbation: eps=1.2, slices=4

### Iteration 55: Dynamic-Perturbation-48 (convnext_small)
- **Loss Formulation:** poly(eps=1.2)
- **Configuration:** Resolution=256x256, Slices=8
- **5-Fold OOF Macro-AUC:** 0.5470
- **Ensemble OOF Macro-AUC:** 0.5035
- **Notes:** Auto-Perturbation: eps=1.2, slices=8

### Phase 5: Grandmaster Self-Distillation Student (convnext_small)
- **Data Expansion:** 58 Gold $\rightarrow$ 308 Studies (Self-Distilled Pseudo-Labels)
- **Student 5-Fold OOF Macro-AUC:** 0.5582
- **Final Ensemble OOF Macro-AUC:** 0.4989
- **Resolution & Slices:** 256x256, K=6
