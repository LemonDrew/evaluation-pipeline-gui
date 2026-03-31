import customtkinter

# CTK Frame to start the evaluation
class EvaluationPage(customtkinter.CTkFrame):

    def __init__(self, master):
        super().__init__(master)

        self.progress = customtkinter.CTkProgressBar(self)
        self.progress.pack(pady=20, padx=20, fill="x")
        self.progress.set(0)

    def run_evaluation(model1, model2):
        return