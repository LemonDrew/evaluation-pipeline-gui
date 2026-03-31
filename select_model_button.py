import customtkinter
from tkinter import filedialog
import os

class SelectModelButton(customtkinter.CTkFrame):

    def __init__(self, master):
        super().__init__(master)

        self.selected_file = None  # store path

        self.grid_columnconfigure(0, weight=1)

        self.button = customtkinter.CTkButton(
            self,
            text="Select Model",
            command=self.select_model
        )
        self.button.grid(row=0, column=0, padx=10, pady=5)

        self.label = customtkinter.CTkLabel(self, text="No file selected")
        self.label.grid(row=1, column=0, padx=10, pady=5)

    def select_model(self):
        file_path = filedialog.askopenfilename(
            title="Select a model",
            filetypes=[
                ("Model files", "*.pt *.onnx *.pth *.tflite"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.selected_file = file_path
            filename = os.path.basename(file_path)
            self.label.configure(text=filename)  # update UI
            print("Selected:", file_path)

    def get(self):
        return self.selected_file