import tkinter as tk
from hwr.app.widgets import *
from hwr.app.pred_interface import ONNETpred


# Overall layout
class App(tk.Frame):
    def __init__(self, root, *args, **kwargs):
        tk.Frame.__init__(self, root, *args, **kwargs)
        text_area = TextArea(root, text="Text", width=50, height=40, padx=3, pady=3)
        pred_area = PredictionArea(root, text_area, text="Correction", width=450, height=50, pady=3)
        draw_area = DrawingArea(root, text_area, pred_area,
                                ONNETpred, text="Writing area", width=450, height=200, padx=3, pady=3)

        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        text_area.grid(row=0, sticky="nsew")
        pred_area.grid(row=1, sticky="ew")
        draw_area.grid(row=2, sticky="ew")

        text_area.grid_rowconfigure(0, weight=1)
        text_area.grid_columnconfigure(0, weight=1)
        pred_area.grid_rowconfigure(0, weight=1)
        draw_area.grid_columnconfigure(0, weight=1)
        draw_area.grid_rowconfigure(0, weight=1)


def main():
    root = tk.Tk()
    root.title('online handwriting recognition')
    root.geometry('{}x{}'.format(1024, 768))
    app = App(root)
    app.grid(sticky="nsew")
    root.mainloop()


if __name__ == "__main__":
    main()
