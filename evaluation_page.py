import customtkinter
import tensorflow as tf
from report_page import ReportPage

# CTK Frame to start the evaluation
class EvaluationPage(customtkinter.CTkFrame):

    status = ["Initialising Models...", "Resizing Test Set", "Running Test", "Generating Report"]

    def __init__(self, master):
        super().__init__(master)

        self.progress = customtkinter.CTkProgressBar(self)
        self.progress.pack(pady=20, padx=20, fill="x")
        self.progress.set(0)

    def run_evaluation(self, model1, model2):

        # Load both models
        first_interpreter = tf.lite.Interpreter(model_path = model1)
        second_interpreter = tf.lite.Interpreter(model_path = model2)

        for widget in self.winfo_children():
            widget.destroy()
        
        self.report_page = ReportPage(self)


    def start_pipeline(self, model1, model2):
        return
    

    def retrieve_minimum_resolution(self, model1,model2):
        return