# Tests Directory

This directory contains **unit tests** for verifying the functionality and stability of each module in the pipeline.

> All tests are written using **pytest** and ensure that the core components  
> of the project (annotation cleaning, YOLO cropping, data augmentation, and classification)  
> behave as expected and maintain reproducibility across updates.

---

## Folder Structure

tests/
├── annotation_cleaner/ # Unit tests for annotation cleaning (SSIM, L1, Edge IoU metrics)
├── yolo_cropper/ # Unit tests for YOLO inference, cropping, and file generation
├── data_augmentor/ # Unit tests for dataset splitting and augmentation logic
├── classifier/ # Unit tests for data loading, training, and evaluation modules
├── utils/ # Unit tests for logging, configuration, and helper utilities

---

## Test Overview

- **Annotation Cleaner Tests** → Verify image cleaning, metric computation, and directory output consistency.  
- **YOLO Cropper Tests** → Validate detection inference, cropping accuracy, and directory structure creation.  
- **Data Augmentor Tests** → Check correct split ratios, augmentation reproducibility, and metadata consistency.  
- **Classifier Tests** → Confirm model loading, training loop integrity, and evaluation metric outputs.  
- **Utils Tests** → Ensure logging and configuration managers work across modules.  
- **Integration Test (`test_main.py`)** → Simulates the entire pipeline to verify end-to-end functionality.

---

## Running Tests

To run all tests:

> pytest -v tests/

To run a specific module’s tests:

> pytest tests/classifier/

---

## Notes

- All tests use **mock data** or **sample datasets** to avoid dependency on private data.  
- Heavy GPU operations (training/inference) are replaced with **lightweight mocks** for speed and reproducibility.  
- Successful test execution ensures that all modules function independently and that the full pipeline operates correctly.
