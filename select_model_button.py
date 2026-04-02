import customtkinter
import tkinter as tk
from tkinter import filedialog
from model_runner import ARCHITECTURES


class SelectModelButton(customtkinter.CTkFrame):
    """
    A composite widget that lets the user pick a model file (.tflite or .pt).
    When a .pt file is chosen, an architecture dropdown appears beneath the button.
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        self._path     = None
        self._arch_var = tk.StringVar(value=list(ARCHITECTURES.keys())[0])

        # ── File picker button ────────────────────────────────────────────────
        self._btn = customtkinter.CTkButton(
            self,
            text="Select Model",
            width=180,
            command=self._pick_file,
        )
        self._btn.grid(row=0, column=0, padx=6, pady=6)

        # ── Architecture dropdown (hidden until a .pt is chosen) ──────────────
        self._arch_label = customtkinter.CTkLabel(self, text="Architecture:")
        self._arch_menu  = customtkinter.CTkOptionMenu(
            self,
            variable=self._arch_var,
            values=list(ARCHITECTURES.keys()),
            width=180,
        )
        # Not gridded yet — shown only for .pt files

    # ── Internal ──────────────────────────────────────────────────────────────

    def _pick_file(self):
        path = filedialog.askopenfilename(
            title="Select model",
            filetypes=[
                ("TFLite models", "*.tflite"),
                ("PyTorch models", "*.pt"),
                ("All files",      "*.*"),
            ],
        )
        if not path:
            return

        self._path = path
        short = path.split("/")[-1]
        if len(short) > 22:
            short = short[:10] + "…" + short[-10:]
        self._btn.configure(text=short)

        if path.endswith(".pt"):
            self._arch_label.grid(row=1, column=0, padx=6, pady=(0, 2))
            self._arch_menu.grid( row=2, column=0, padx=6, pady=(0, 6))
        else:
            self._arch_label.grid_remove()
            self._arch_menu.grid_remove()


    def get(self) -> dict | None:
        """
        Returns {"path": str, "arch": str | None} or None if no file chosen.
        arch is only set for .pt files.
        """
        if not self._path:
            return None
        arch = self._arch_var.get() if self._path.endswith(".pt") else None
        return {"path": self._path, "arch": arch}

    def reset(self):
        self._path = None
        self._btn.configure(text="Select Model")
        self._arch_label.grid_remove()
        self._arch_menu.grid_remove()