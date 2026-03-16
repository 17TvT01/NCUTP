import customtkinter as ctk
from tkinter import ttk

class ResultTree(ctk.CTkFrame):
    def __init__(self, master, on_item_click_cb=None, **kwargs):
        super().__init__(master, border_width=1, border_color="#555555", fg_color="#2b2b2b", **kwargs)
        self.on_item_click_cb = on_item_click_cb
        
        ctk.CTkLabel(self, text="Kết quả", text_color="#aaaaaa", font=("Segoe UI", 12)).pack(anchor="w", padx=10, pady=(5, 0))
        res_content = ctk.CTkFrame(self, fg_color="transparent")
        res_content.pack(fill="both", expand=True, padx=10, pady=(2, 10))
        
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="#2b2b2b", foreground="#cccccc", fieldbackground="#2b2b2b", borderwidth=0)
        style.configure("Treeview.Heading", background="#333333", foreground="white", borderwidth=0)
        style.map("Treeview", background=[("selected", "#1f538d")])

        self.tree = ttk.Treeview(res_content, columns=("ID", "Voxel", "Z", "Y", "X", "Malig", "Color"), show="headings", height=15)
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=50, anchor="center")
            
        self.tree.pack(fill="both", expand=True)
        self.tree.bind('<<TreeviewSelect>>', self._on_tree_select)

    def _on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel: return
        vals = self.tree.item(sel[0])["values"]
        if vals and self.on_item_click_cb:
            z_str = str(vals[2])
            if '-' in z_str:
                z_str = z_str.split('-')[0]
            try:
                z_idx = int(z_str)
                self.on_item_click_cb(z_idx)
            except ValueError:
                pass

    def clear(self):
        [self.tree.delete(i) for i in self.tree.get_children()]

    def add_item(self, values):
        self.tree.insert("", "end", values=values)
