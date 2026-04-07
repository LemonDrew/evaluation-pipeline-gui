import os
import threading
import customtkinter
import tkinter as tk
import torch
from torchvision.ops import box_iou, box_convert
from PIL import Image
from model_runner import ModelRunner


class EvaluationPage(customtkinter.CTkFrame):
    """
    Runs the evaluation pipeline in a background thread and updates the UI
    via self.after() to keep the main loop responsive.

    Pipeline:
        1. Load both models
        2. Load images + labels from dataset folder
        3. Resize every image to the minimum resolution
        4. Run inference on both models, recording time and predicted label
        5. Compute Top-1 accuracy, mean inference time, throughput
        6. Hand results to ReportPage
    """

    _STEPS = [
        "Loading models…",
        "Loading dataset…",
        "Running evaluation…",
        "Generating report…",
    ]

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.grid_columnconfigure(0, weight=1)

        # ── Status label ──────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Initialising…")
        self._status_lbl = customtkinter.CTkLabel(
            self, textvariable=self._status_var, font=("", 14)
        )
        self._status_lbl.grid(row=0, column=0, padx=20, pady=(30, 6))

        self._progress = customtkinter.CTkProgressBar(self)
        self._progress.grid(row=1, column=0, padx=20, pady=6, sticky="ew")
        self._progress.set(0)

        self._sub_var = tk.StringVar(value="")
        self._sub_lbl = customtkinter.CTkLabel(
            self, textvariable=self._sub_var, font=("", 11), text_color="gray"
        )
        self._sub_lbl.grid(row=2, column=0, padx=20, pady=(0, 20))

    def run_evaluation(self, model1: dict, model2: dict, dataset_folder: str, label : str, is_live: bool):
        """
        Start the evaluation pipeline in a background thread.

        Args:
            model1 / model2:  dicts with keys "path" and "arch" (arch=None for tflite)
            dataset_folder:   root folder containing images + a labels file
            is_live: boolean value indicating whether live camera is on
        """
        self._model1_info    = model1
        self._model2_info    = model2
        self._dataset_folder = dataset_folder
        self._expected_label = label

        print(f"Model 1: {model1}")
        print(f"Model 2: {model2}")
        print(f"Dataset folder: {dataset_folder}")

        # If no live camera preview, then we use a dataset
        if (not is_live):   
            t = threading.Thread(target=self._pipeline, daemon=True)
            t.start()
   

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, text: str, progress: float, sub: str = ""):
        self.after(0, lambda: self._status_var.set(text))
        self.after(0, lambda: self._progress.set(progress))
        self.after(0, lambda: self._sub_var.set(sub))

    # Helper Function to load models
    def _loadModels(self):
        self._set_status(self._STEPS[0], 0.05)
        runner1 = ModelRunner(self._model1_info["path"], self._model1_info.get("arch"))
        runner2 = ModelRunner(self._model2_info["path"], self._model2_info.get("arch"))
        runner1.load()
        runner2.load()
        return runner1, runner2

    # Helper Function to load images from folder
    def _loadImages(self):
        self._set_status(self._STEPS[1], 0.25)

        images_dir = os.path.join(self._dataset_folder, "images")
        labels_dir = os.path.join(self._dataset_folder, "labels")

        image_files = sorted(os.listdir(images_dir))

        dataset = []
        labels  = []

        for img_file in image_files:
            # Skip non-image files
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            label_filename = os.path.splitext(img_file)[0] + ".txt"
            label_path     = os.path.join(labels_dir, label_filename)

            if not os.path.exists(label_path):
                print(f"Warning: No label found for {img_file}, skipping.")
                continue

            img_path = os.path.join(images_dir, img_file)
            dataset.append((img_path, self._expected_label))
            labels.append(label_path)

        return dataset, labels
        
    def _runInference(self, runner1, runner2, dataset):

        self._set_status(self._STEPS[2], 0.35)

        # Retrieve size of each model
        h1, w1 = runner1.input_size() 
        h2, w2 = runner2.input_size()  

        results1, results2 = [], []
        correct1 = correct2 = 0
        n = len(dataset)

        for i, (img_path, true_label) in enumerate(dataset):

            image1 = Image.open(img_path).convert("RGB").resize((w1, h1))
            image2 = Image.open(img_path).convert("RGB").resize((w2, h2))

            r1 = runner1.predict(image1)
            r2 = runner2.predict(image2)

            results1.append(r1)
            results2.append(r2)

            if r1["label"] == true_label:
                correct1 += 1
            if r2["label"] == true_label:
                correct2 += 1

            progress = 0.35 + 0.55 * ((i + 1) / n)
            self._set_status(
                self._STEPS[3],
                progress,
                f"Image {i + 1} / {n}",
            )

        runner1.close()
        runner2.close()

        return results1, results2, correct1, correct2
    
    def _processResults(self, results1, results2, correct1, correct2, labels):

        iou_scores1 = []
        iou_scores2 = []
        
        # Calculates IoU
        for i, lbl_path in enumerate(labels):
            with open(lbl_path, "r") as f:
                parts = f.readline().strip().split()
                _, cx, cy, w, h = parts
                gt_box = torch.tensor([[float(cx), float(cy), float(w), float(h)]])
                gt_box_xyxy = box_convert(gt_box, in_fmt="cxcywh", out_fmt="xyxy")

            # Model 1 IoU
            pred_box1 = results1[i].get("box")
            if pred_box1:
                b1 = torch.tensor([[pred_box1["cx"], pred_box1["cy"], pred_box1["w"], pred_box1["h"]]])
                b1_xyxy = box_convert(b1, in_fmt="cxcywh", out_fmt="xyxy")
                iou1 = box_iou(gt_box_xyxy, b1_xyxy)[0][0].item()
                iou_scores1.append(iou1)

            # Model 2 IoU
            pred_box2 = results2[i].get("box")
            if pred_box2:
                b2 = torch.tensor([[pred_box2["cx"], pred_box2["cy"], pred_box2["w"], pred_box2["h"]]])
                b2_xyxy = box_convert(b2, in_fmt="cxcywh", out_fmt="xyxy")
                iou2 = box_iou(gt_box_xyxy, b2_xyxy)[0][0].item()
                iou_scores2.append(iou2)

        self._set_status(self._STEPS[3], 0.95)

        def summarise(results, correct, iou_scores):
            times   = [r["time_ms"] for r in results]
            mean_ms = sum(times) / len(times)
            total_s = sum(times) / 1000
            return {
                "recall":           correct / len(results),
                "mean_iou":         sum(iou_scores) / len(iou_scores) if iou_scores else 0.0,
                "mean_time_ms":     mean_ms,
                "throughput_img_s": len(results) / total_s,
                "per_image":        results,
            }

        metrics1 = summarise(results1, correct1, iou_scores1)
        metrics2 = summarise(results2, correct2, iou_scores2)

        print("Metrics for Model 1:")
        for k, v in metrics1.items():
            if k != "per_image":
                print(f"  {k}: {v}")
        print("Metrics for Model 2:")
        for k, v in metrics2.items():
            if k != "per_image":
                print(f"  {k}: {v}")

        return metrics1, metrics2

    # Main Evaluation Pipeline to be run 
    def _pipeline(self):
        try:
            
            # Load Models
            runner1, runner2 = self._loadModels()

            # Load Images
            dataset, labels = self._loadImages()

            # Run Inference
            results1, results2, correct1, correct2 = self._runInference(runner1, runner2, dataset)

            # Process Results
            metrics1, metrics2 = self._processResults(results1, results2, correct1, correct2, labels)

            # # Hand off to ReportPage on main thread
            self.after(0, lambda: self._show_report(metrics1, metrics2))

        except Exception as exc:
            self.after(0, lambda exc = exc: self._show_error(str(exc)))

    def _show_report(self, metrics1: dict, metrics2: dict):
        from report_page import ReportPage

        for widget in self.master.winfo_children():
            widget.destroy()

        report = ReportPage(
            self.master,
            model1_name=self._model1_info["path"].split("/")[-1],
            model2_name=self._model2_info["path"].split("/")[-1],
            metrics1=metrics1,
            metrics2=metrics2,
        )
        report.pack(fill="both", expand=True)

    def _show_error(self, message: str):
        self._set_status(f"Error: {message}", 0, "")
        err_lbl = customtkinter.CTkLabel(
            self, text=message, text_color="red", wraplength=500
        )
        err_lbl.grid(row=3, column=0, padx=20, pady=10)