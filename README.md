# Crop and Conquer: A Dual-Pipeline Framework for Trustworthy Visual Classification

> **Crop and Conquer: A Dual-Pipeline Framework for Trustworthy Visual Classification**  
> This repository provides the **official open-source implementation** of the paper *"Crop and Conquer: A Dual-Pipeline Framework for Trustworthy Visual Classification"*.  
> The study was conducted and authored by **Kyeongha Hwang (Suwon University, Korea)**, who carried out all experiments and analysis.

---

## Reproduction Guide

This section describes how to reproduce and execute the provided dual-pipeline framework.

### Environment Setup

Run the following command to automatically create the environment and install all dependencies:

```bash
bash setup.sh
```

This script will:

- Create a new Conda environment (tf_env)

- Install all Python dependencies listed in requirements.txt

- Clone external repositories (Darknet for YOLOv2/v4 and YOLOv5) into the third_party/ directory

- Download pretrained YOLOv2 and YOLOv4 weights

- Download fine-tuned YOLOv8s weights from Google Drive (these are custom-trained weights used in the project for reproducible evaluation)

- After setup, activate the environment using:

```bash
conda activate tf_env
```

#### Gemini API Key Configuration

> The Annotation Cleaning module uses the Google Gemini API for removing human-drawn annotations via generative inpainting.
Before running the pipeline, set your Gemini API key as an environment variable:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

The utils/config.yaml automatically loads this key through the following line:
```bash
api_key: ${GEMINI_API_KEY}
```

### Run the Main Pipeline

Once the setup is complete, execute the unified dual-pipeline using:

```bash
python src/main.py
```

This command launches the Crop-and-Conquer pipeline, consisting of:

1. Annotation Cleaning — Removes human-drawn annotation marks using a generative model.

2. YOLO-based Cropping — Detects and crops damage regions via YOLO (v2, v4, v5, v8).

3. Data Augmentation — Balances the dataset using class-aware augmentation.

4. Classification & Evaluation — Trains CNN-based classifiers and measures performance.

#### CLI Options
The main script supports several command-line options for flexible execution:

| Option        | Description   |
| ------------- | ------------- |
| --annot_clean [on/off]  | Performs annotation cleaning to remove human-drawn marks from original images. |
| --test [on/off]  | Runs the pipeline in test mode, processing only a few images (default: 3) to avoid unnecessary API costs.  |
| --yolo_crop [on/off]  | Enables ROI cropping using a fine-tuned YOLO model. |
| --yolo_model [model_name]  | Specifies which YOLO model to use for cropping. Available: yolov2, yolov4, yolov5, yolov8s, yolov8m, yolov8l, yolov8x.  |

### Example Usage

```bash
# Run annotation cleaning only
python src/main.py --annot_clean on --yolo_crop off

# Run YOLO-based cropping only
python src/main.py --annot_clean off --yolo_crop on --yolo_model yolov8m

# Test mode (limited to 3 images)
python src/main.py --annot_clean on --test on
```

### Configuration Control

All module settings and directory paths are managed via:
```bash
utils/config.yaml
```

This configuration file centrally controls every module:
- AnnotationCleaner
- YOLOCropper
- DataAugmentor
- Classifier

Since the full dataset cannot be shared publicly, training and evaluation cannot be executed end-to-end.
However, you can customize input paths in utils/config.yaml to apply the implemented pipeline on your own dataset.

> AnnotationCleaner Input Directory
> - The folder data/sample/annotation_cleaner/only_annotation_image/ is used to store images for the annotation cleaning process.
> - This design helps prevent unnecessary API costs by allowing selective cleaning instead of processing the entire dataset.
> - In the provided sample, all test images are included in this folder by default.
> - If you wish to perform annotation cleaning on your own images, simply place them inside this directory before running:
```bash
python src/main.py --annot_clean on
```

### Project Overview

- Main entry point: src/main.py

- Config file: utils/config.yaml

- Logs: stored automatically under logs/

- Third-party repos: cloned into third_party/ (Darknet, YOLOv5)

- Model checkpoints: saved under checkpoints/ and saved_model/


## Note on Data Privacy
The dataset used in this study is private and cannot be distributed publicly.
Therefore, only a small sample dataset is provided under:

```bash
data/sample/
```

- The sample data allows you to test the annotation cleaning and cropping functionalities.
- Training and evaluation are disabled by default to prevent runtime errors due to missing private data.

---

## Example Results (Sample Dataset)

- This example illustrates how image data transforms through each stage of the pipeline.
- YOLO Only: ROI-cropped images from the original dataset.
- Gen: Annotation-cleaned images generated from the original dataset.
- Gen + YOLO: ROI-cropped images after the annotation cleaning process.
- These four datasets are then used as inputs for the Classifier, enabling quantitative analysis of performance variations across each experimental setting.

| Category | (a) Original | (b) YOLO Only | (c) Gen Only | (d) Gen + YOLO |
|:---------:|:---------:|:--------------:|:------------:|:----------------:|
| **Repair** | <img src="data/sample/original/repair/img_01.png" width="250"> | <img src="data/sample/original_crop/yolov8s/repair/img_01.png" width="250"> | <img src="data/sample/generation/repair/img_01.png" width="250"> | <img src="data/sample/generation_crop/yolov8s/repair/img_01.png" width="250"> |
|            | <img src="data/sample/original/repair/img_02.jpg" width="250"> | <img src="data/sample/original_crop/yolov8s/repair/img_02.jpg" width="250"> | <img src="data/sample/generation/repair/img_02.jpg" width="250"> | <img src="data/sample/generation_crop/yolov8s/repair/img_02.jpg" width="250"> |
|           | <img src="data/sample/original/repair/img_03.png" width="250"> | <img src="data/sample/original_crop/yolov8s/repair/img_03.jpg" width="250"> | <img src="data/sample/generation/repair/img_03.png" width="250"> | <img src="data/sample/generation_crop/yolov8s/repair/img_03.jpg" width="250"> |
| **Replace** | <img src="data/sample/original/replace/img_01.jpg" width="250"> | <img src="data/sample/original_crop/yolov8s/replace/img_01.jpg" width="250"> | <img src="data/sample/generation/replace/img_01.jpg" width="250"> | <img src="data/sample/generation_crop/yolov8s/replace/img_01.jpg" width="250"> |
|           |  <img src="data/sample/original/replace/img_02.jpg" width="250"> | <img src="data/sample/original_crop/yolov8s/replace/img_02.jpg" width="250"> | <img src="data/sample/generation/replace/img_02.jpg" width="250"> | <img src="data/sample/generation_crop/yolov8s/replace/img_02.jpg" width="250"> |
|           | <img src="data/sample/original/replace/img_03.jpg" width="250"> | <img src="data/sample/original_crop/yolov8s/replace/img_03.jpg" width="250"> | <img src="data/sample/generation/replace/img_03.jpg" width="250"> | <img src="data/sample/generation_crop/yolov8s/replace/img_03.jpg" width="250"> |


---

## Experimental Results

> The following section presents the comparison and analysis of classification performance trained on the four datasets — (a) Original, (b) YOLO Only, (c) Gen Only, and (d) Gen + YOLO — described above.

---

### Grad-CAM Visualization

The Grad-CAM analysis illustrates the regions of interest (ROIs) that the classifier focuses on under each experimental setup.

| (a) Original | (b) YOLO Only | (c) Gen Only | (d) Gen + YOLO |
|---------------|---------------|---------------|----------------|
| ![gradcam_orig_01](assets/Grad-CAM/original/img_01.png) | ![gradcam_orig_crop_01](assets/Grad-CAM/original/img_01_crop.png) | ![gradcam_gen_01](assets/Grad-CAM/generation/img_01.png) | ![gradcam_gen_crop_01](assets/Grad-CAM/generation/img_01_crop.png) |
| ![gradcam_orig_02](assets/Grad-CAM/original/img_02.png) | ![gradcam_orig_crop_02](assets/Grad-CAM/original/img_02_crop.png) | ![gradcam_gen_02](assets/Grad-CAM/generation/img_02.png) | ![gradcam_gen_crop_02](assets/Grad-CAM/generation/img_02_crop.png) |

> *Observation:*  
> When the model trained on the YOLO Cropped dataset was used to infer the original images,
it exhibited much stronger attention around the actual damage regions. 
> This phenomenon was consistently observed in both the Original and Generatively Cleaned datasets.
> We argue that this improvement arises because YOLO cropping effectively removes contextless, non-damage regions,
allowing the model to focus more precisely on the true damage areas during learning.

---

### Classification Accuracy & Data Reliability Analysis

> This section presents the quantitative comparison of classification performance and dataset reliability across the four experimental settings.
- Each configuration is evaluated using the best-performing model checkpoint obtained from ten independent runs.
- In this study, Data Reliability is defined as (\( 1 - \text{(Bias Ratio)} \)), where bias refers to human annotations or artificial artifacts such as hand markings, needles, or other non-damage elements included in the images.

| Condition | Data Reliability | Annotation Clean | YOLO Crop | Best Acc (%) |
|------------|------------------|------------------|------------|---------------|
| (a) Original | 66% | ✗ | ✗ | **95.39** |
| (b) YOLO Only | 89.80% | ✗ | ✓ | **97.46** |
| (c) Gen Only | 100% | ✓ | ✗ | **88.94** |
| (d) Gen + YOLO | 100% | ✓ | ✓ | **93.40** |

> *Observation:*  
> - Applying YOLO Cropping resulted in accuracy improvement across both the Original and Generatively Cleaned datasets. 
> - By cropping only the true damage regions inside human annotations, YOLO-based preprocessing achieved up to +23.80% higher data reliability compared to the Original dataset 
> - Combining **annotation cleaning** and **YOLO cropping** yields more interpretable models with higher spatial precision,  
> - Although Generative AI–based annotation cleaning completely removes human markings (reaching 100% data reliability),
this process is non-reproducible and occasionally alters the damage patterns, indicating a need for further research.

---

For citation, experimental details, and additional documentation, please refer to the paper:
“Crop and Conquer: A Dual-Pipeline Framework for Trustworthy Visual Classification” (Hwang, K., Suwon University).