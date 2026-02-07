#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict

from utils.load_config import load_yaml_config
from utils.logging import get_logger

from src.annotation_cleaner.evaluation import metrics as M
from src.annotation_cleaner.evaluation.generate_anchor import run_anchor_generation
from src.annotation_cleaner.evaluation.visualization import plot_metrics_scatter

logger = get_logger("Evaluator")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

class Evaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.eval_conf = config["annotation_cleaner"]["evaluate"]
        
        # Paths
        dirs = self.eval_conf["dirs"]
        self.dir_ref = Path(dirs["baseline_crop"])       # Ground Truth (Clean)
        self.dir_anchors = Path(dirs["anchor_root"])     # Root of anchors
        self.out_dir = Path(dirs["results"])
        
        # Weights
        self.vgg_path = self.eval_conf.get("weights", {}).get("vgg", None)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Metric Model (VGG)
        logger.info(f"Initializing VGG Feature Extractor on {self.device}...")
        self.feat_model = M.VGGFeatureExtractor(weight_path=self.vgg_path, device=self.device)

    def run(self):
        """
        Executes the full evaluation pipeline:
        1. Generate Anchors (Data Prep)
        2. Calculate Metrics (Evaluation)
        3. Visualize Results (Reporting)
        """
        logger.info("=" * 60)
        logger.info(" PIPELINE START: Annotation Cleaner Evaluation")
        logger.info("=" * 60)

        # [Step 1] Trigger Anchor Generation (Data Preparation)
        logger.info(">> [Step 1/3] Generating Evaluation Data (Anchors)...")
        try:
            run_anchor_generation(self.config)
        except Exception as e:
            logger.error(f"Anchor generation failed. Aborting evaluation. Error: {e}")
            return

        # [Step 2] Metric Calculation
        logger.info(">> [Step 2/3] Calculating Metrics...")
        
        if not self.dir_ref.exists():
            logger.error(f"Reference directory missing: {self.dir_ref}")
            return

        ref_files = sorted([p for p in self.dir_ref.glob("*") if p.suffix.lower() in IMG_EXTS])
        if not ref_files:
            logger.error("No reference images found for evaluation.")
            return

        # Discover Anchor Types
        if not self.dir_anchors.exists():
            logger.error(f"Anchor root missing: {self.dir_anchors}")
            return

        anchor_types = sorted([d.name for d in self.dir_anchors.iterdir() if d.is_dir()])
        logger.info(f"Detected Anchors: {anchor_types}")

        # Map files
        anchor_maps = {}
        for atype in anchor_types:
            adir = self.dir_anchors / atype
            anchor_maps[atype] = {p.stem: p for p in adir.glob("*") if p.suffix.lower() in IMG_EXTS}

        # Calculation Loop
        all_data = []
        for ref_path in tqdm(ref_files, desc="Eval Images"):
            stem = ref_path.stem
            filename = ref_path.name
            
            img_ref = self._safe_imread(ref_path)
            if img_ref is None: continue
            
            t_ref = M.cv2_to_tensor(img_ref, self.device)

            for atype in anchor_types:
                anchor_file = anchor_maps[atype].get(stem)
                if not anchor_file: continue
                
                img_anchor = self._safe_imread(anchor_file)
                if img_anchor is None: continue

                # Resize if needed
                if img_anchor.shape != img_ref.shape:
                    img_anchor = cv2.resize(img_anchor, (img_ref.shape[1], img_ref.shape[0]))

                # Metrics
                l1 = M.calculate_l1_distance(img_ref, img_anchor)
                psnr = M.calculate_psnr(img_ref, img_anchor)
                ssim = M.calculate_ssim(img_ref, img_anchor, self.device)
                edge_l1 = M.calculate_edge_distance(img_ref, img_anchor)
                
                t_anchor = M.cv2_to_tensor(img_anchor, self.device)
                with torch.no_grad():
                    d_feat = M.calc_feature_distance(self.feat_model, t_ref, t_anchor)
                
                # Save Vis (Edge Map) - Optional: Limit to first few to save time? Currently saves all.
                self._save_visualization(img_ref, img_anchor, atype, filename)

                all_data.append({
                    "Filename": filename,
                    "Anchor": atype,
                    "L1": l1,
                    "Edge_L1": edge_l1,
                    "D_FeatMap": d_feat,
                    "PSNR": psnr,
                    "SSIM": ssim
                })

        # Save Results
        if not all_data:
            logger.error("No metrics computed. Check if anchor filenames match reference filenames.")
            return

        self.out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_data)
        df.to_csv(self.out_dir / "metrics_details.csv", index=False)
        
        summary = df.groupby("Anchor")[["L1", "Edge_L1", "D_FeatMap", "PSNR", "SSIM"]].mean()
        summary.to_csv(self.out_dir / "metrics_summary.csv")
        
        self._print_report(summary)

        # ---------------------------------------------------------------------
        # [Step 3] Visualization (PDF Generation)
        # ---------------------------------------------------------------------
        logger.info(">> [Step 3/3] Generating Figures...")
        plot_metrics_scatter(self.config)

        logger.info("=" * 60)
        logger.info(" PIPELINE FINISHED SUCCESSFULLY")
        logger.info("=" * 60)

    def _safe_imread(self, path):
        try:
            data = np.fromfile(str(path), np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except:
            return None

    def _save_visualization(self, img_ref, img_anchor, anchor_type, filename):
        vis_dir = self.out_dir / "visualizations" / filename
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        def get_edge(img):
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.Canny(g, 100, 200)

        cv2.imwrite(str(vis_dir / "clean.png"), get_edge(img_ref))
        cv2.imwrite(str(vis_dir / f"{anchor_type}.png"), get_edge(img_anchor))

    def _print_report(self, summary_df):
        print("\n" + "="*60)
        print(" FINAL EVALUATION REPORT")
        print("="*60)
        metrics_conf = [
            ("L1", False), ("Edge_L1", False), ("D_FeatMap", False),
            ("PSNR", True), ("SSIM", True)
        ]
        for metric, higher_better in metrics_conf:
            direction = "(↑ Higher is Better)" if higher_better else "(↓ Lower is Better)"
            print(f"\n[{metric}] {direction}")
            if metric not in summary_df.columns: continue
            vals = summary_df[metric].sort_values(ascending=not higher_better)
            rank = 1
            for anchor, score in vals.items():
                mark = " [OURS]" if anchor.lower() == "gac" else ""
                print(f"  {rank}. {anchor:<10} : {score:.4f}{mark}")
                rank += 1
        print("\n" + "="*60)

def run_evaluation(config_path="config.yaml"):
    # Load config here if running directly
    config = load_yaml_config(config_path)
    Evaluator(config).run()

if __name__ == "__main__":
    # Assuming run from project root
    run_evaluation("utils/config.yaml")