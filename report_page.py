import customtkinter
import csv

class ReportPage(customtkinter.CTkFrame):

    def __init__(self, master):
        super().__init__(master)

        self.data = [
            {"name" : "Nill", "branch" : "South", "Year" : "2002", "cgpa" : "1.2"}
        ]

        self.report_button = customtkinter.CTkButton(self, text="View Report")
        self.report_button.pack(pady=20, padx=20, fill="x")


    def save_to_csv(self):
        with open('university_records.csv', 'w', newline='') as csvfile:
            fieldnames = ['name', 'branch', 'year', 'cgpa']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.data)
        return