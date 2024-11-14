import tkinter as tk
from tkinter import messagebox
from unlearning_algorithm import LLMUnlearning
from unlearning_script import load_data

class UnlearningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Unlearning Interface")

        self.model_name = "bert-base-uncased"
        self.unlearning = LLMUnlearning(self.model_name)

        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self.root, text="Enter data to unlearn:")
        self.label.pack()

        self.text_entry = tk.Entry(self.root, width=50)
        self.text_entry.pack()

        self.unlearn_button = tk.Button(self.root, text="Unlearn", command=self.unlearn_data)
        self.unlearn_button.pack()

        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack()

    def unlearn_data(self):
        data_to_unlearn = self.text_entry.get()
        if not data_to_unlearn:
            messagebox.showerror("Error", "Please enter data to unlearn.")
            return

        # Load data to unlearn
        data_to_unlearn = [{"text": data_to_unlearn, "labels": 0}]  # Example data format
        self.unlearning.gradient_descent_unlearning(data_to_unlearn)

        self.result_label.config(text="Unlearning process completed.")

if __name__ == "__main__":
    root = tk.Tk()
    app = UnlearningApp(root)
    root.mainloop()
