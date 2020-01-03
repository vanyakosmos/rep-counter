import functools
import json
from pathlib import Path

import socket

from flask import Flask, jsonify, request

from nn.consts import IMG_ROOT, LABELS_FILE


app = Flask(__name__, template_folder=".", static_folder=IMG_ROOT)


class Labeler:
    def __init__(self):
        self.images = list(IMG_ROOT.glob("*.png"))
        self._index = 0
        self._labels = {}
        self.load_checkpoint()

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        index = max(0, min(self.max_index, index))
        self._index = index

    @property
    def max_index(self):
        return min(len(self._labels), len(self.images) - 1)

    @property
    def labels(self):
        return sorted(self._labels.values(), key=lambda e: e["index"])

    def add_image_label(self, label):
        image = self.get_image()
        self._labels[self.index] = {
            "index": self.index,
            "image": image,
            "label": label,
        }
        self.index += 1
        self.save_checkpoint()

    def get_image(self, index=None):
        fp = self.images[index or self.index]
        filename = str(fp.parts[-1])
        return filename

    def save_checkpoint(self):
        LABELS_FILE.write_text(json.dumps(self.labels))

    def get_labels(self) -> dict:
        if LABELS_FILE.exists():
            data = json.loads(LABELS_FILE.read_text())
            return {item["index"]: item for item in data}
        return {}

    def load_checkpoint(self):
        self._labels = self.get_labels()
        self.index = len(self._labels)

    def serialize(self):
        return {
            "image": self.get_image(),
            "labels": self.labels,
            "index": self.index,
            "maxIndex": self.max_index,
            "nextImage": self.get_image(self.max_index),
        }


labeler = Labeler()


def action(url, method="post"):
    route = app.route(url, methods=[method.upper()])

    def dec(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            return jsonify(labeler.serialize())

        return route(wrapper)

    return dec


@app.route("/")
def index_view():
    root = Path(app.root_path)
    return root.joinpath("index.html").read_text()


@action("/state", method="get")
def state_view():
    pass


@action("/move")
def move_view():
    d = request.json
    if "stride" in d:
        labeler.index += d["stride"]
    elif "index" in d:
        labeler.index = d["index"]


@action("/label/<label>")
def label_view(label):
    label = int(label)
    labeler.add_image_label(label)


def main():
    host = socket.gethostbyname(socket.gethostname())
    port = 5000

    app.run(host=host, port=port, debug=True)


if __name__ == "__main__":
    main()
