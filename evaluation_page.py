import os
import threading
import customtkinter
import tkinter as tk
from PIL import Image
from model_runner import ModelRunner


class EvaluationPage(customtkinter.CTkFrame):
    """
    Runs the evaluation pipeline in a background thread and updates the UI
    via self.after() to keep the main loop responsive.

    Pipeline:
        1. Load both models
        2. Determine minimum shared input resolution
        3. Load images + labels from dataset folder
        4. Resize every image to the minimum resolution
        5. Run inference on both models, recording time and predicted label
        6. Compute Top-1 accuracy, mean inference time, throughput
        7. Hand results to ReportPage
    """

    _STEPS = [
        "Loading models…",
        "Resolving minimum resolution…",
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

    def run_evaluation(self, model1: dict, model2: dict, dataset_folder: str):
        """
        Start the evaluation pipeline in a background thread.

        Args:
            model1 / model2:  dicts with keys "path" and "arch" (arch=None for tflite)
            dataset_folder:   root folder containing images + a labels file
        """
        self._model1_info    = model1
        self._model2_info    = model2
        self._dataset_folder = dataset_folder

        t = threading.Thread(target=self._pipeline, daemon=True)
        t.start()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, text: str, progress: float, sub: str = ""):
        self.after(0, lambda: self._status_var.set(text))
        self.after(0, lambda: self._progress.set(progress))
        self.after(0, lambda: self._sub_var.set(sub))

    def _find_labels_file(self) -> str | None:
        for fname in os.listdir(self._dataset_folder):
            if fname.endswith(".txt"):
                return os.path.join(self._dataset_folder, fname)
        return None

    def _load_dataset(self, labels_path: str) -> list[tuple[str, int]]:
        """
        Returns [(image_path, label_int), …] sorted by filename.
        Labels file: one integer label per line, matched to sorted image filenames.
        """
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = sorted(
            f for f in os.listdir(self._dataset_folder)
            if os.path.splitext(f)[1].lower() in image_exts
        )

        with open(labels_path, "r") as fh:
            raw = [line.strip() for line in fh if line.strip()]

        labels = [int(l) for l in raw]

        if len(labels) != len(image_files):
            raise ValueError(
                f"Label count ({len(labels)}) != image count ({len(image_files)})"
            )

        return [
            (os.path.join(self._dataset_folder, img), lbl)
            for img, lbl in zip(image_files, labels)
        ]

    @staticmethod
    def retrieve_minimum_resolution(runner1: ModelRunner, runner2: ModelRunner) -> tuple[int, int]:
        """Return the smallest (H, W) accepted by both models."""
        h1, w1 = runner1.input_size()
        h2, w2 = runner2.input_size()
        return min(h1, h2), min(w1, w2)

    # Main Evaluation Pipeline to be run 
    def _pipeline(self):
        try:
            # Step 1 — load models
            self._set_status(self._STEPS[0], 0.05)
            runner1 = ModelRunner(self._model1_info["path"], self._model1_info.get("arch"))
            runner2 = ModelRunner(self._model2_info["path"], self._model2_info.get("arch"))
            runner1.load()
            runner2.load()

            # Step 2 — minimum resolution
            self._set_status(self._STEPS[1], 0.15)
            h, w = self.retrieve_minimum_resolution(runner1, runner2)

            # Step 3 — load dataset
            self._set_status(self._STEPS[2], 0.25)
            labels_path = self._find_labels_file()
            if labels_path is None:
                raise FileNotFoundError("No .txt labels file found in dataset folder.")
            dataset = self._load_dataset(labels_path)
            n = len(dataset)

            # Step 4 — run inference
            self._set_status(self._STEPS[3], 0.35)

            results1, results2 = [], []
            correct1 = correct2 = 0

            for i, (img_path, true_label) in enumerate(dataset):
                image = Image.open(img_path).convert("RGB").resize((w, h))

                r1 = runner1.predict(image)
                r2 = runner2.predict(image)

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

            # Step 5 — compute metrics
            self._set_status(self._STEPS[4], 0.95)

            def summarise(results: list[dict], correct: int) -> dict:
                times   = [r["time_ms"] for r in results]
                mean_ms = sum(times) / len(times)
                total_s = sum(times) / 1000
                return {
                    "accuracy":         correct / n,
                    "mean_time_ms":     mean_ms,
                    "throughput_img_s": n / total_s,
                    "per_image":        results,
                }

            metrics1 = summarise(results1, correct1)
            metrics2 = summarise(results2, correct2)

            # Hand off to ReportPage on main thread
            self.after(0, lambda: self._show_report(metrics1, metrics2))

        except Exception as exc:
            self.after(0, lambda: self._show_error(str(exc)))

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