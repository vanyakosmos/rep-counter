import json
import tkinter as tk
from functools import partial

from nn.consts import IMG_ROOT, LABELS_FILE


def center(win):
    win.update_idletasks()
    width = win.winfo_width()
    height = win.winfo_height()
    x = (win.winfo_screenwidth() // 2) - (width // 2)
    y = (win.winfo_screenheight() // 2) - (height // 2)
    win.geometry("{}x{}+{}+{}".format(width, height, x, y))


class App(tk.Tk):
    def __init__(self, images):
        super().__init__()
        # validation
        assert images
        # attrs setup
        self.images = images
        self._index = 0
        self.labels = {}
        # buttons
        tk.Button(self, text="Save", width=20, command=self.save_checkpoint).grid(row=0, column=0)
        tk.Button(self, text="Load Checkpoint", width=20, command=self.load_checkpoint).grid(row=0, column=1)
        tk.Button(self, text="Save and Quit", width=20, command=self.save_and_quit).grid(row=0, column=2)
        # pic label
        self.picture_label = tk.Label(self)
        self.picture_label.grid(row=1, column=0, columnspan=3)
        # log label
        self.log_label = tk.Label(self, font=("Courier", 12), width=60, justify=tk.LEFT)
        self.log_label.grid(row=1, column=4, sticky="nw")
        # keyboard bindings
        self.bind("<Left>", partial(self.show_next_image, stride=-1))
        self.bind("<Right>", partial(self.show_next_image, stride=1))
        self.bind("j", partial(self.add_image_label, mark=0))
        self.bind("k", partial(self.add_image_label, mark=1))
        self.bind("l", partial(self.add_image_label, mark=2))
        # data setup
        self.render_image()
        self.render_labels()

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        index = max(0, min(len(self.labels), len(self.images) - 1, index))
        self._index = index

    def add_image_label(self, event, mark):
        filename = self.images[self.index]
        self.labels[str(filename)] = mark
        self.index += 1
        self.render()

    def show_next_image(self, event, stride):
        self.index += stride
        self.render()

    def save_checkpoint(self, event=None):
        LABELS_FILE.write_text(json.dumps(self.labels))

    def load_checkpoint(self, event=None):
        if LABELS_FILE.exists():
            self.labels = json.loads(LABELS_FILE.read_text())
            self.index = len(self.labels)
            self.render()

    def save_and_quit(self, event=None):
        self.save_checkpoint()
        self.destroy()

    def render_labels(self):
        lines = [f"{k} - {v}" for k, v in self.labels.items()]
        lines = lines[: self.index][-10:]
        text = "\n".join(lines)
        text = f"j - relaxed, k - tense, l - unfocused\n{text}"
        self.log_label.config(text=text)

    def render_image(self):
        filename = self.images[self.index]
        image = tk.PhotoImage(file=filename)
        self.picture_label.image = image  # keep reference to the image
        self.picture_label.config(image=image)
        self.title(filename)

    def render(self):
        self.render_image()
        self.render_labels()

    def run(self):
        self.mainloop()


def main():
    images = list(IMG_ROOT.glob("*.png"))
    app = App(images)
    center(app)
    app.run()


if __name__ == "__main__":
    main()
