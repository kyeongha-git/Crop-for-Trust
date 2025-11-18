# Utils Directory

This directory contains utility scripts for **configuration management**, **logging**, and **global control**  
across all modules in the AI pipeline.

> **Note:**  
> The `utils` package enables centralized configuration and logging so that  
> users can control the entire pipeline behavior through a single configuration file (`config.yaml`).

---

## Folder Structure

utils/
├── config_manager.py # Dynamically manages config paths and synchronizes modules
├── config.yaml # Global configuration file controlling all modules
├── load_config.py # Loads and parses YAML config for use across the pipeline
└── logging.py # Unified logging system for runtime monitoring

---

## Description

### 1️⃣ config.yaml
- The **core configuration file** of the entire project.  
- All experiment parameters, paths, and module settings are defined here.  
- Users only need to modify values in this file (especially main paths) to control the entire pipeline.

---

### 2️⃣ load_config.py
- Responsible for reading and loading the YAML configuration (`config.yaml`).  
- Provides easy access to parameters within each module (e.g., annotation_cleaner, yolo_cropper, classifier).  
- Ensures consistent parameter parsing across all scripts.

---

### 3️⃣ logging.py
- Implements a **unified logging system** used across all modules.  
- Automatically generates log files with timestamps and saves them under `logs/`.  
- Tracks training progress, evaluation results, and runtime exceptions.

---

### 4️⃣ config_manager.py
- Handles **dynamic configuration management** at the top-level `main.py`.  
- When the user specifies the main project path or mode, this script automatically updates  
  the relevant module paths (annotation_cleaner, yolo_cropper, classifier, etc.).  
- Enables **modular coordination** — ensuring all submodules use synchronized configurations.

---

## Summary

- The `utils/` directory provides the **foundation** for project-wide control and logging.  
- Users only need to modify **`config.yaml` once** to adapt the entire pipeline to a new dataset or environment.  
- This design ensures **reproducibility**, **consistency**, and **ease of configuration** across all components.