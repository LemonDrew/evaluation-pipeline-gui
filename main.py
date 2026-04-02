import tkinter as tk
from tkinter import filedialog
import customtkinter
from select_model_button import SelectModelButton
from evaluation_page import EvaluationPage


def build_main_ui(root: customtkinter.CTk):

    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)

    # Variable to store the dataset
    dataset_var = tk.StringVar(value="") 
    is_live_camera = tk.BooleanVar(value=False)

    # Frame to contain the dataset button
    top_frame = customtkinter.CTkFrame(root, fg_color="transparent")
    top_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
    top_frame.grid_columnconfigure(0, weight=1)
    top_frame.grid_columnconfigure(1, weight=1)
    top_frame.grid_columnconfigure(2, weight=1)

    # Helper function to select the dataset
    def select_dataset():
        folder = filedialog.askdirectory(title="Select dataset folder")
        if folder:
            dataset_var.set(folder)
            short = folder.split("/")[-1] or folder
            dataset_btn.configure(text=f"{short}")

    # Button for Selecting Dataset
    dataset_btn = customtkinter.CTkButton(
        top_frame, text="Select Dataset", width=180, command=select_dataset
    )
    dataset_btn.grid(row=0, column=1, pady=(10, 0))
    # Button's label
    dataset_lbl = customtkinter.CTkLabel(
        top_frame, textvariable=dataset_var, text_color="gray", anchor="w"
    )
    dataset_lbl.grid(row=1, column=1, sticky="ew")

    # ── Model selection frame ─────────────────────────────────────────────────
    model_selection_frame = customtkinter.CTkFrame(root)
    model_selection_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
    model_selection_frame.grid_columnconfigure(0, weight=1)
    model_selection_frame.grid_columnconfigure(1, weight=1)

    first_model_button = SelectModelButton(model_selection_frame)
    first_model_button.grid(row=0, column=0, padx=10, pady=10)

    second_model_button = SelectModelButton(model_selection_frame)
    second_model_button.grid(row=0, column=1, padx=10, pady=10)

    # ── Error label ───────────────────────────────────────────────────────────
    error_var = tk.StringVar(value="")
    error_lbl = customtkinter.CTkLabel(root, textvariable=error_var, text_color="red")
    error_lbl.grid(row=2, column=0)

    def start_evaluation():
        model1  = first_model_button.get()
        model2  = second_model_button.get()
        dataset = dataset_var.get()

        missing = []
        if not model1:
            missing.append("Model 1")
        if not model2:
            missing.append("Model 2")
        if not dataset:
            missing.append("Dataset folder")

        if missing:
            error_var.set(f"Please select: {', '.join(missing)}")
            return

        error_var.set("")

        # ── Navigate to Evaluation Page ──────────────────────────────────────
        for widget in root.winfo_children():
            widget.destroy()

        eval_page = EvaluationPage(root)
        eval_page.pack(fill="both", expand=True)
        eval_page.run_evaluation(model1, model2, dataset)

    def set_camera_preview():
        status = not is_live_camera.get()
        is_live_camera.set(status)
        print("Status", status)

    # Checkbox for enabling/disabling live camera
    live_camera_btn = customtkinter.CTkCheckBox(
        root, text="Enable Live Camera Preview", command=set_camera_preview
    )
    live_camera_btn.grid(row=3, column=0, pady=(5, 5))

    start_btn = customtkinter.CTkButton(
        root, text="Start Evaluation", command=start_evaluation
    )
    start_btn.grid(row=4, column=0, padx=10, pady=10, sticky="ew")

    

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Evaluation Pipeline")
        self.geometry("800x440")
        build_main_ui(self)


if __name__ == "__main__":
    app = App()
    app.mainloop()