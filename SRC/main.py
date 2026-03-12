import customtkinter as ctk
from ui.analysis_tab import AnalysisTab
from ui.training_tab import TrainingTab

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class LungNoduleApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Lung Nodule Assistant")
        self.geometry("1100x750")
        self.minsize(900, 600)
        self.after(100, lambda: self.state('zoomed'))  # Delay maximize sau khi UI render xong
        
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=15, pady=0)
        
        self.tab_analysis = self.tabview.add("Phân tích")
        self.tab_train = self.tabview.add("Huấn luyện AI")
        
        self.analysis_ui = AnalysisTab(self.tab_analysis)
        self.analysis_ui.pack(fill="both", expand=True)
        
        self.train_ui = TrainingTab(self.tab_train)
        self.train_ui.pack(fill="both", expand=True)

if __name__ == "__main__":
    app = LungNoduleApp()
    app.mainloop()
