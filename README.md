# CBCT Tooth Segmentation Pipeline

## Overview
End-to-end pipeline for 3D CBCT dental segmentation using nnU-Net v2 with pretrained weights.
Accepts .mha, .nii, .nii.gz, and DICOM inputs. Outputs per-tooth segmentation masks with
an interactive HTML viewer for clinical visualization.

## Dataset
- Primary: ToothFairy2 (Dataset112, 480 volumes)
- Format: .mha volumes with FDI tooth labels
- Split: Inference demonstrated on 3 held-out test volumes

## Model
- Framework: nnU-Net v2 (self-configuring medical segmentation)
- Weights: DentalSegmentator v1.0.0 pretrained checkpoint (Zenodo: 10829675)
- Architecture: 3D full-resolution U-Net, auto-configured for dental CBCT
- Device: CUDA (Tesla T4)

## Preprocessing
- Format conversion: .mha to .nii.gz via SimpleITK
- Resampling: isotropic spacing (handled by nnU-Net internally)
- Normalization: HU clipping and z-score normalization (nnU-Net default)

## Results
| Patient | Dice Score |
|---------|-----------|
| ToothFairy2F_001 | 0.7301 |
| ToothFairy2F_002 | 0.7177 |
| ToothFairy2F_003 | 0.7275 |
| **Mean** | **0.7251** |

## Visualization
Self-contained HTML viewer (viewer.html) built with Niivue WebGL library.
Features:
- Axial/coronal/sagittal slice scrolling
- Segmentation mask toggle (on/off)
- Opacity control slider
- 3D surface rendering toggle

## How to Run

### Requirements
```
pip install nnunetv2 nibabel SimpleITK scipy
```

### Inference
```
python inference.py --input /path/to/scan.nii.gz --output /path/to/output/
```

### Visualization
Place scan.nii.gz and mask.nii.gz in the same folder as viewer.html.
Open a terminal in that folder and run:
```
python -m http.server 8080
```
Then open http://localhost:8080/viewer.html in your browser.

## Repository Structure
```
cbct-tooth-segmentation/
  inference.py        # main inference script
  preprocess.py       # preprocessing utilities
  requirements.txt    # dependencies
  Dockerfile          # reproducibility
  viewer.html         # interactive visualization
  scan.nii.gz         # example input scan
  mask.nii.gz         # example predicted mask
  README.md           # this file
```

## References
- nnU-Net: Isensee et al., Nature Methods 2021
- DentalSegmentator: Zenodo record 10829675
- ToothFairy2: https://ditto.ing.unimore.it/toothfairy2
