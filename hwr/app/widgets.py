import tkinter as tk
from tkinter import scrolledtext


# the drawing pad
class DrawingArea(tk.LabelFrame):
    def __init__(self, parent, text_area, predict_area, pred, **kwargs):
        super().__init__(parent, **kwargs)
        self.text_area = text_area
        self.predict_area = predict_area
        self.pred = pred()
        # state variables
        self.btn1pressed = False
        self.newline = True
        self.xorig = None
        self.yorig = None
        self.drawing = False
        self.after_list = []
        self.curr_stroke = 0
        self.points = []

        self.setup_canvas()

    def mouse1press(self, event):
        if not self.drawing:
            self.text_area.set_word_start()
        self.drawing = True
        self.btn1pressed = True
        self.xorig = event.x
        self.yorig = event.y
        self.curr_stroke += 1
        self.points.append([])
        if self.after_list:
            self.after_cancel(self.after_list.pop(0))

    def mouse1release(self, event):
        self.btn1pressed = False
        self.xorig = None
        self.yorig = None
        # Wait 2s, if mouse1 was not pressed, clear canvas
        after_id = self.after(1000, self.clear_canvas)
        self.after_list.append(after_id)

    def mousemove(self, event):
        # left click held down
        if self.btn1pressed:
            if self.xorig:
                event.widget.create_line(self.xorig, self.yorig, event.x, event.y,
                                         smooth=tk.TRUE, width=3)
            self.xorig = event.x
            self.yorig = event.y
            self.points[-1].append((event.x, event.y))

    def setup_canvas(self):
        self.canvas = tk.Canvas(self, bg="white")
        self.canvas.bind("<Motion>", self.mousemove)
        self.canvas.bind("<ButtonPress-1>", self.mouse1press)
        self.canvas.bind("<ButtonRelease-1>", self.mouse1release)
        self.canvas.grid(column=0, row=0, sticky="nsew")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.drawing = False
        self.get_predictions()
        self.points = []

    def get_predictions(self):
        features = self.pred.get_features(self.points)
        result = self.pred.predict(features, 5)
        self.text_area.insert_text(result[0])
        self.text_area.set_word_end()
        self.predict_area.update_buttons(result)


# The predicted text
class TextArea(tk.LabelFrame):

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.setup_textbox()

    def setup_textbox(self):
        self.textbox = tk.scrolledtext.ScrolledText(self)
        self.textbox.grid(row=0, column=0, sticky="nsew")
        self.textbox.tag_configure("TAG", background="#e9e9e9")
        self.set_word_start()
        self.set_word_end()

    def get_input(self):
        print(self.textbox.get(1.0, tk.END))

    def insert_text(self, text):
        self.textbox.tag_remove(tk.SEL, 1.0, tk.END)
        self.textbox.delete("WORDSTART", "WORDEND")
        self.textbox.insert("WORDSTART", text, (tk.SEL,))

    # Set the current ending mark as start mark
    def set_word_start(self):
        self.textbox.mark_set("WORDSTART", tk.INSERT)
        self.textbox.mark_gravity("WORDSTART", tk.LEFT)

    def set_word_end(self):
        self.textbox.mark_set("WORDEND", tk.INSERT)


# the top-n predictions for correction
class PredictionArea(tk.LabelFrame):
    def __init__(self, parent, text_area, **kwargs):
        super().__init__(parent, **kwargs)
        self.buttons = []
        self.setup_predictions(5)
        self.text_area = text_area

    def setup_predictions(self, n):
        for i in range(n):
            b = tk.Button(self)
            b.grid(row=0, column=i, sticky="nsew")
            self.buttons.append(b)
            self.grid_columnconfigure(i, weight=1)

    def update_buttons(self, preds):
        assert (len(preds) == len(self.buttons))
        for i in range(len(self.buttons)):
            self.buttons[i].config(text=preds[i], command=lambda t=preds[i]: self.text_area.insert_text(t))

