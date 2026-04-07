import csv
import tkinter as tk
import customtkinter
from tkinter import filedialog
 
 
class ReportPage(customtkinter.CTkFrame):
    """
    Displays evaluation results for two models side-by-side:
        • Top-1 accuracy
        • Mean inference time per image (ms)
        • Total throughput (images/sec)
    Also provides a CSV export button.
    """
 
    def __init__(
        self,
        master,
        model1_name: str,
        model2_name: str,
        metrics1: dict,
        metrics2: dict,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self._model1_name = model1_name
        self._model2_name = model2_name
        self._metrics1    = metrics1
        self._metrics2    = metrics2
 
        self.grid_columnconfigure(0, weight=1)
        self._build()
 
    def _build(self):
        title = customtkinter.CTkLabel(
            self, text="Evaluation Report", font=("", 20, "bold")
        )
        title.grid(row=0, column=0, pady=(24, 4))
 
        subtitle = customtkinter.CTkLabel(
            self,
            text=f"{self._model1_name}  vs  {self._model2_name}",
            font=("", 12),
            text_color="gray",
        )
        subtitle.grid(row=1, column=0, pady=(0, 20))
 
        # Metrics table frame
        table_frame = customtkinter.CTkFrame(self)
        table_frame.grid(row=2, column=0, padx=30, pady=10, sticky="ew")
        self.grid_columnconfigure(0, weight=1)

        # Headers for table
        headers = ["Metric", self._model1_name, self._model2_name, "Winner"]
        col_widths = [220, 160, 160, 120]
 
        # Header row
        for col, (text, width) in enumerate(zip(headers, col_widths)):
            lbl = customtkinter.CTkLabel(
                table_frame,
                text=text,
                font=("", 13, "bold"),
                width=width,
                anchor="center",
            )
            lbl.grid(row=0, column=col, padx=8, pady=8)
 
        # Divider
        sep = customtkinter.CTkFrame(table_frame, height=2, fg_color="gray")
        sep.grid(row=1, column=0, columnspan=4, sticky="ew", padx=6)
 
        rows = self._build_rows()
        for r_idx, (label, val1, val2, winner) in enumerate(rows, start=2):
            bg = ("gray95", "gray15") if r_idx % 2 == 0 else ("white", "gray20")
            for col, text in enumerate([label, val1, val2, winner]):
                cell = customtkinter.CTkLabel(
                    table_frame,
                    text=text,
                    width=col_widths[col],
                    anchor="center",
                    fg_color=bg,
                    corner_radius=4,
                )
                cell.grid(row=r_idx, column=col, padx=8, pady=4)
 
        # Buttons row
        btn_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=3, column=0, pady=24)
 
        export_btn = customtkinter.CTkButton(
            btn_frame, text="Export CSV", width=160, command=self._export_csv
        )
        export_btn.grid(row=0, column=0, padx=12)
 
        back_btn = customtkinter.CTkButton(
            btn_frame,
            text="New Evaluation",
            width=160,
            fg_color="gray",
            command=self._restart,
        )
        back_btn.grid(row=0, column=1, padx=12)
 
    def _build_rows(self) -> list[tuple]:
        """Return list of (metric_name, val1_str, val2_str, winner_str)."""
        m1, m2 = self._metrics1, self._metrics2
        rows = []
 
        # Accuracy — higher is better
        a1, a2 = m1["recall"], m2["recall"]
        rows.append((
            "Top-1 Accuracy",
            f"{a1 * 100:.2f}%",
            f"{a2 * 100:.2f}%",
            self._winner(a1, a2, higher_better=True),
        ))
 
        # Mean inference time — lower is better
        t1, t2 = m1["mean_time_ms"], m2["mean_time_ms"]
        rows.append((
            "Mean Inference Time (ms)",
            f"{t1:.3f} ms",
            f"{t2:.3f} ms",
            self._winner(t1, t2, higher_better=False),
        ))
 
        # Throughput — higher is better
        tp1, tp2 = m1["throughput_img_s"], m2["throughput_img_s"]
        rows.append((
            "Throughput (img/s)",
            f"{tp1:.1f}",
            f"{tp2:.1f}",
            self._winner(tp1, tp2, higher_better=True),
        ))
 
        return rows
 
    @staticmethod
    def _winner(v1: float, v2: float, higher_better: bool) -> str:
        if abs(v1 - v2) < 1e-9:
            return "Tie"
        if higher_better:
            return "Model 1 ✓" if v1 > v2 else "Model 2 ✓"
        else:
            return "Model 1 ✓" if v1 < v2 else "Model 2 ✓"
 
    # ── CSV export ────────────────────────────────────────────────────────────
 
    def _export_csv(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save evaluation results",
            initialfile="evaluation_report.csv",
        )
        if not path:
            return
 
        n = len(self._metrics1["per_image"])
 
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
 
            # Summary section
            writer.writerow(["=== SUMMARY ==="])
            writer.writerow(["Metric", self._model1_name, self._model2_name])
            writer.writerow([
                "Top-1 Accuracy (%)",
                f"{self._metrics1['recall'] * 100:.4f}",
                f"{self._metrics2['recall'] * 100:.4f}",
            ])
            writer.writerow([
                "Mean Inference Time (ms)",
                f"{self._metrics1['mean_time_ms']:.4f}",
                f"{self._metrics2['mean_time_ms']:.4f}",
            ])
            writer.writerow([
                "Throughput (img/s)",
                f"{self._metrics1['throughput_img_s']:.4f}",
                f"{self._metrics2['throughput_img_s']:.4f}",
            ])
            writer.writerow([])
 
            # Per-image section
            writer.writerow(["=== PER-IMAGE RESULTS ==="])
            writer.writerow([
                "Image Index",
                f"{self._model1_name} — Predicted Label",
                f"{self._model1_name} — Time (ms)",
                f"{self._model2_name} — Predicted Label",
                f"{self._model2_name} — Time (ms)",
            ])
            for i in range(n):
                r1 = self._metrics1["per_image"][i]
                r2 = self._metrics2["per_image"][i]
                writer.writerow([
                    i,
                    r1["label"],
                    f"{r1['time_ms']:.4f}",
                    r2["label"],
                    f"{r2['time_ms']:.4f}",
                ])
 
        # Brief confirmation
        ok_lbl = customtkinter.CTkLabel(
            self, text=f"✓ Saved to {path.split('/')[-1]}", text_color="green"
        )
        ok_lbl.grid(row=4, column=0, pady=(0, 10))
 
    # ── Restart ───────────────────────────────────────────────────────────────
 
    def _restart(self):
        """Destroy report and rebuild the main App UI."""
        # Walk up to the root CTk window and restart it
        root = self.master
        for widget in root.winfo_children():
            widget.destroy()
        from main import build_main_ui
        build_main_ui(root)