import tkinter as tk
from tkinter import scrolledtext

from hwr.app.event import Event
from hwr.app.pubsub import pub


# the drawing pad
class WritingPadView(tk.LabelFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        # state variables
        self.btn1pressed = False
        self.newline = True
        self.xorig = None
        self.yorig = None
        self.drawing = False
        self.after_list = []
        self.curr_stroke = 0
        self.points = []

        # View
        self.setup_canvas()

    def mouse1press(self, event):
        if not self.drawing:
            pub(Event.START_DRAWING, None)

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
        after_id = self.after(1000, self.on_end_drawing)
        self.after_list.append(after_id)

    def mousemove(self, event):
        # left click held down
        if self.btn1pressed:
            if self.xorig:
                event.widget.create_line(self.xorig, self.yorig, event.x, event.y,
                                         smooth=tk.TRUE, width=3)
            self.xorig = event.x
            self.yorig = event.y
            # append point to last stroke
            self.points[-1].append((event.x, event.y))

    def setup_canvas(self):
        self.canvas = tk.Canvas(self, bg="white")
        self.canvas.bind("<Motion>", self.mousemove)
        self.canvas.bind("<ButtonPress-1>", self.mouse1press)
        self.canvas.bind("<ButtonRelease-1>", self.mouse1release)
        self.canvas.grid(column=0, row=0, sticky="nsew")

    def on_end_drawing(self):
        self.clear_canvas()
        pub(Event.END_DRAWING, self.points)
        self.points = []

    def clear_canvas(self):
        self.canvas.delete("all")
        self.drawing = False


class PredictedTextView(tk.LabelFrame):

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.setup_textbox()

    def on_predictions_setted(self, preds):
        self.insert_text(preds[0])
        self.set_word_end()

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
class CorrectionsView(tk.LabelFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.buttons = []
        self.setup_predictions(5)

    def setup_predictions(self, n):
        for i in range(n):
            b = tk.Button(self)
            b.grid(row=0, column=i, sticky="nsew")
            self.buttons.append(b)
            self.grid_columnconfigure(i, weight=1)

    def update_buttons(self, preds):
        assert (len(preds) == len(self.buttons))
        for i in range(len(self.buttons)):
            self.buttons[i].config(text=preds[i], command=lambda t=preds[i]: pub(Event.PRED_SELECTED, t))

