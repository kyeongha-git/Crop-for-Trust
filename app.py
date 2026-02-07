import gradio as gr
import subprocess
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

# ======================================================
# Configuration
# ======================================================
ROOT_DIR = Path("/app")
SCRIPT_PATH = ROOT_DIR / "run_full_reproduction.sh"
SANDBOX_ROOT = ROOT_DIR / "data/demo_run"

# [User Selection Maps]
REPLACE_SAMPLES = {
    "🔄 Replace Case 1 (Circle Crack)":       "img_01.jpg",
    "🔄 Replace Case 2 (Big Crack)":   "img_02.jpg",
    "🔄 Replace Case 3 (Star Crack)":   "img_03.jpg",
}

REPAIR_SAMPLES = {
    "🛠️ Repair Case 1": "img_01.png",
    "🛠️ Repair Case 2": "img_02.jpg",
    "🛠️ Repair Case 3":    "img_03.jpg",
}

def find_result_image(base_dir: Path, class_name: str, target_filename: str) -> Optional[str]:
    search_dir = base_dir / class_name
    if not search_dir.exists(): search_dir = base_dir 
    if not search_dir.exists(): return None

    target_stem = Path(target_filename).stem
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for f in search_dir.rglob("*"): 
        if f.suffix.lower() in valid_extensions and target_stem in f.name:
            return str(f)
    return None

def compute_full_diff(img_path1: str, img_path2: str, save_path: str) -> Optional[str]:
    """Diff 생성"""
    try:
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        if img1 is None or img2 is None: return None
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        diff = cv2.absdiff(img1, img2)
        diff = cv2.multiply(diff, np.array([3.0]))
        cv2.imwrite(save_path, diff)
        return save_path
    except Exception:
        return None

def run_pipeline(model_name: str, replace_selection: str, repair_selection: str) -> Tuple[str, ...]:
    
    replace_file = REPLACE_SAMPLES[replace_selection]
    repair_file = REPAIR_SAMPLES[repair_selection]
    
    command = [
        "/bin/bash", str(SCRIPT_PATH), 
        "--yolo_model", model_name,
        "--repair_image", repair_file,
        "--replace_image", replace_file
    ]
    
    try:
        print(f"[INFO] Running Optimized Pipeline...")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(ROOT_DIR))
        stdout, stderr = process.communicate()
        full_log = f"--- STDOUT ---\n{stdout}\n\n--- STDERR ---\n{stderr}"
        
        if process.returncode != 0:
            return f"Pipeline Failed (Code {process.returncode})\n\n{full_log}", None, None, None, None

        vis_class = "replace"
        vis_file = replace_file
        
        path_orig = find_result_image(SANDBOX_ROOT / "original", vis_class, vis_file)
        path_yolo = find_result_image(SANDBOX_ROOT / "generation_crop" / model_name, vis_class, vis_file)
        path_gen = find_result_image(SANDBOX_ROOT / "generation", vis_class, vis_file)
        
        path_diff = None
        if path_orig and path_gen:
            diff_name = f"diff_{Path(vis_file).stem}.jpg"
            save_dest = SANDBOX_ROOT / "generation" / diff_name
            path_diff = compute_full_diff(path_orig, path_gen, str(save_dest))

        # [Added] Disclaimer Note in Logs
        disclaimer_note = (
            "\n⚠️ Note: This result was generated in real-time using Gemini 2.5 Flash API.\n"
            "Due to the stochastic nature of Generative AI, details in the restoration "
            "may vary slightly from previous runs or expected ground truth.\n"
        )

        success_msg = (
            f"Pipeline Completed Successfully!\n"
            f"Optimization Active: Repair(GenAI Skip) + Replace(GenAI Run)\n"
            f"Visualizing: {vis_class}/{vis_file}\n"
            f"{disclaimer_note}\n"
            f"{full_log}"
        )

        return success_msg, path_orig, path_yolo, path_gen, path_diff

    except Exception as e:
        return f"System Error: {str(e)}", None, None, None, None


# ======================================================
# UI Layout
# ======================================================
# [Modified] Dark Mode Compatible Disclaimer
description = """
<div style="text-align: center;">
    <h1>Crop for Trust: Reliability-Aware Visual Debiasing Pipeline</h1>
    <p>
        Runs the full pipeline (GenAI → YOLO → Classification).<br>
        <b>Optimization:</b> 'Repair' skips GenAI to reduce latency, while 'Replace' runs fully to verify restoration.
    </p>
    
    <div style="
        background-color: #2b2b2b; 
        border: 1px solid #f0ad4e; 
        color: #e0e0e0; 
        padding: 15px; 
        border-radius: 8px; 
        margin: 20px auto; 
        font-size: 0.95em;
        text-align: left;
        max-width: 800px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    ">
        <h4 style="margin: 0 0 10px 0; color: #f0ad4e; font-weight: bold;">⚠️ Disclaimer</h4>
        This system utilizes the <b>Gemini 2.5 Flash API</b> for image in-painting. 
        Due to the generative nature of the model, 
        <b>output results may vary between executions and might occasionally contain artifacts or hallucinations.</b> 
        This is an expected behavior of current GenAI technologies.
    </div>
</div>
"""

with gr.Blocks(title="Crop for Trust Demo", theme=gr.themes.Soft()) as demo:
    gr.HTML(description)

    with gr.Row(variant="panel"):
        with gr.Column():
            replace_dropdown = gr.Dropdown(
                choices=list(REPLACE_SAMPLES.keys()), value=list(REPLACE_SAMPLES.keys())[0],
                label="1. Replace Sample (GenAI RUN)"
            )
        with gr.Column():
            repair_dropdown = gr.Dropdown(
                choices=list(REPAIR_SAMPLES.keys()), value=list(REPAIR_SAMPLES.keys())[0],
                label="2. Repair Sample (GenAI SKIP)"
            )
        with gr.Column():
            model_dropdown = gr.Dropdown(
                choices=["yolov5", "yolov8s", "yolov8m"], value="yolov5",
                label="3. YOLO Model"
            )

    run_btn = gr.Button("🚀 Run Pipeline", variant="primary")
    
    with gr.Row():
        log_output = gr.Textbox(label="Logs", lines=20)
        with gr.Column():
            with gr.Row():
                img_orig = gr.Image(label="Original Input")
                img_gen = gr.Image(label="GenAI Restoration")
            with gr.Row():
                img_yolo = gr.Image(label="YOLO Crop")
                img_diff = gr.Image(label="Diff Analysis")

    run_btn.click(
        fn=run_pipeline,
        inputs=[model_dropdown, replace_dropdown, repair_dropdown],
        outputs=[log_output, img_orig, img_yolo, img_gen, img_diff]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)