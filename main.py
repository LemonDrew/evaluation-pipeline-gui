import customtkinter
from select_model_button import SelectModelButton

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Evaluation Pipeline")
        self.geometry("800x400")

        # Grid config
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)  # frame expands

        # Dataset button
        self.dataset_button = customtkinter.CTkButton(
            self, text="Select Dataset", command=self.select_dataset
        )
        self.dataset_button.grid(row=0, column=0, padx=10, pady=10)

        # Frame
        self.checkbox_frame = customtkinter.CTkFrame(self)
        self.checkbox_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.checkbox_frame.grid_columnconfigure(0, weight=1)
        self.checkbox_frame.grid_columnconfigure(1, weight=1)

        # Model buttons
        self.first_model_button = SelectModelButton(self.checkbox_frame)
        self.first_model_button.grid(row=0, column=0, padx=10, pady=10)

        self.second_model_button = SelectModelButton(self.checkbox_frame)
        self.second_model_button.grid(row=0, column=1, padx=10, pady=10)

        # Start button
        self.button = customtkinter.CTkButton(
            self, text="Start Evaluation", command=self.start_evaluation
        )
        self.button.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

    def select_dataset(self):
        print("Selecting dataset")

    def start_evaluation(self):
        print("Starting test")

app = App()
app.mainloop()