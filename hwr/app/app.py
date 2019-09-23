from hwr.app.model import ONNETpred, Model
from hwr.app.views import *
import tkinter as tk
from hwr.app.pubsub import sub
from hwr.app.event import Event


# Overall layout
class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        super(App, self).__init__(*args, **kwargs)

        self.title('On-line handwriting recognition')
        self.geometry('{}x{}'.format(1024, 768))

        model = Model(ONNETpred)
        text_area = PredictedTextView(self, text="Text", width=50, height=40, padx=3, pady=3)
        pred_area = CorrectionsView(self, text="Correction", width=450, height=50, pady=3)
        draw_area = WritingPadView(self, text="Writing area", width=450, height=200, padx=3, pady=3)

        # Events
        # Text area
        sub(Event.PRED_SELECTED, lambda x: text_area.insert_text(x))
        sub(Event.START_DRAWING, lambda x: text_area.set_word_start())
        sub(Event.PRED_SETTED, lambda x: text_area.on_predictions_setted(x))
        # Correction area
        sub(Event.PRED_SETTED, lambda x: pred_area.update_buttons(x))
        # Model
        sub(Event.END_DRAWING, lambda x: model.compute_predictions(x))

        # Relative layout of the child views
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        text_area.grid(row=0, sticky="nsew")
        pred_area.grid(row=1, sticky="ew")
        draw_area.grid(row=2, sticky="ew")
        text_area.grid_rowconfigure(0, weight=1)
        text_area.grid_columnconfigure(0, weight=1)
        pred_area.grid_rowconfigure(0, weight=1)
        draw_area.grid_columnconfigure(0, weight=1)
        draw_area.grid_rowconfigure(0, weight=1)

    def run(self):
        self.mainloop()


def main():
    App().run()


if __name__ == "__main__":
    main()
