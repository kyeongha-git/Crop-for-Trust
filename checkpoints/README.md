# Checkpoints Directory

This directory stores **model checkpoint files** (`.pt`) generated during training.

> **Note:**  
> The training dataset used in this project is **private**, and thus  
> checkpoint weights (`last.pt`, `best.pt`) are **not publicly released**.  
> During training, these files are automatically saved in the corresponding subdirectories.

---

## Folder Structure

checkpoints/
├── yolo_cropper/ # YOLO-based cropping model weights
└── classifier/ # CNN-based classification model weights

---

## Description

- **yolo_cropper/** → Contains checkpoints (`last.pt`, `best.pt`) for YOLO-based detection models.  
- **classifier/** → Contains checkpoints for CNN-based classification models  
  (e.g., MobileNetV2, ResNet, VGG, ViT) during training.  

---

**Note:**  
These weights are not publicly shared, but this directory structure is preserved  
to ensure **reproducibility** and **pipeline consistency**.  
When training is executed, checkpoint files will be automatically saved here.