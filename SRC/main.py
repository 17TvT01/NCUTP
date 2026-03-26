import customtkinter as ctk
from ui.analysis_tab import AnalysisTab
from ui.training_tab import TrainingTab
from ui.compare_tab import CompareTab

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class LungNoduleApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Lung Nodule Assistant")
        self.geometry("1100x750")
        self.minsize(900, 600)
        
        def maximize_window():
            self.state('zoomed')
            
        # Tăng thời gian chờ lên để CustomTkinter hoàn tất các tính toán giao diện bên trong
        # trước khi ép hệ điều hành phóng to cửa sổ
        self.after(300, maximize_window)
        
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=15, pady=0)
        
        self.tab_analysis = self.tabview.add("Phân tích")
        self.tab_train = self.tabview.add("Huấn luyện AI")
        self.tab_compare = self.tabview.add("So sánh ảnh")

        self.compare_ui = CompareTab(self.tab_compare)
        self.compare_ui.pack(fill="both", expand=True)
        
        self.analysis_ui = AnalysisTab(
            self.tab_analysis,
            on_images_loaded=self.compare_ui.set_source_images,
            on_results_ready=self.compare_ui.set_analysis_results,
            on_clusters_ready=self.compare_ui.set_clusters,
            on_fill_changed=self.compare_ui.set_fill_enabled,
        )
        self.analysis_ui.pack(fill="both", expand=True)
        
        self.train_ui = TrainingTab(self.tab_train)
        self.train_ui.pack(fill="both", expand=True)

        self.bind_all("<Left>", self._on_left_key, add="+")
        self.bind_all("<Right>", self._on_right_key, add="+")

    def _on_left_key(self, event):
        self._step_active_tab_slice(-1)

    def _on_right_key(self, event):
        self._step_active_tab_slice(1)

    def _step_active_tab_slice(self, delta):
        active_tab = self.tabview.get()
        if active_tab == "Phân tích":
            self.analysis_ui.step_slice(delta)
        elif active_tab == "So sánh ảnh":
            self.compare_ui.step_slice(delta)

if __name__ == "__main__":
    app = LungNoduleApp()
    app.mainloop()
