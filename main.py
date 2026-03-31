from tkinterdnd2 import DND_FILES, TkinterDnD
import customtkinter

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Evaluation Pipeline")
        self.geometry("800x400")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Widget to select the dataset
        self.dataset_button = customtkinter.CTkButton(self, text="Select Dataset", command=self.select_dataset)
        self.dataset_button.grid(row=0, column=0, padx=10, pady=10)
        
        # Frame to contain the 2 checkboxes
        self.checkbox_frame = customtkinter.CTkFrame(self)
        self.checkbox_frame.grid(row=1, column=0, padx=10, pady=(10, 0), stick="nsew")
        self.checkbox_frame.grid_columnconfigure(0, weight=1)
        self.checkbox_frame.grid_columnconfigure(1, weight=1)
        self.checkbox_frame.grid_rowconfigure(0, weight=1)

        # First File Widget to accept the first model
        self.button = customtkinter.CTkButton(self.checkbox_frame, text="Select First Model for evaluation", command=self.choose_first_model)
        self.button.grid(row=0, column=0, padx=10, pady=10)

        # Second File Widget to accept the second model
        self.button = customtkinter.CTkButton(self.checkbox_frame, text="Select Second Model for evaluation", command=self.start_evaluation)
        self.button.grid(row=0, column=1, padx=10, pady=10)

        self.button = customtkinter.CTkButton(self, text="Start Evaluation", command=self.start_evaluation)
        self.button.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

    def select_dataset(self):
        print("Selecting dataset")

    def choose_first_model(self):
        print("Selecting first model")

    def choose_second_model(self):
        print("Selecting second model")

    def start_evaluation(self):
        print("Starting test")

app = App()
app.mainloop()