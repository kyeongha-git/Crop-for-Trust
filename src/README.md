# Source Code Directory (`src/`)

This directory contains the core implementation of the AI pipeline,  
divided into modular components for annotation cleaning, YOLO-based cropping,  
data augmentation, and classification.

---

## Folder Overview

src/
├── annotation_cleaner/ # Removes human annotations using generative AI
├── yolo_cropper/ # Detects and crops damage regions using YOLO models
├── data_augmentor/ # Splits and augments datasets
├── classifier/ # Trains and evaluates CNN-based classification models
└── main.py # Unified pipeline entry point

---

Each subdirectory contains its own source code, configuration logic, and execution scripts.  
Please refer to the **root README** for details on how to execute the pipeline.

---

## Code Flow & Experimental Pipeline

The following diagram summarizes the overall code flow and experimental process used in the paper:

<p align="center"> <img src="../assets/flowchart/figure.png" width="800"> </p>

Figure:
- The Crop-and-Conquer framework consists of four sequential yet modular stages
> Annotation Cleaning, YOLO-based Cropping, Data Augmentation, and Classification.
> each of which can be executed independently or as part of the unified main.py pipeline.

