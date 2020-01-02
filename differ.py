from typing import Tuple

import cv2
import numpy as np
from keras import Model
from keras.engine.saving import load_model
from skimage.metrics import structural_similarity

from nn.model import IM_HEIGHT, IM_WIDTH


Arr = np.ndarray


class Differ:
    def __init__(self, multichannel=False):
        self.multichannel = multichannel

    def diff(self, a: Arr, b: Arr) -> Tuple[Arr, float]:
        """
        Should return diff matrix with value between 0 and 1,
        where 1 means that entity has the same representation in A nad B
        """
        raise NotImplementedError

    def compare(self, a: Arr, b: Arr) -> Tuple[Arr, float]:
        if not self.multichannel:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        d, score = self.diff(a, b)
        d = (d * 255).astype(np.uint8)
        if not self.multichannel:
            d = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
        return d, score


class SSIMDiffer(Differ):
    def diff(self, a: Arr, b: Arr) -> Tuple[Arr, float]:
        score, diff = structural_similarity(a, b, full=True, multichannel=self.multichannel)
        diff = (diff + 1) / 2
        return diff, score


class RMSDiffer(Differ):
    def __init__(self, multichannel=False, dilate=3, threshold=25):
        super().__init__(multichannel=multichannel)
        self.dilate = dilate
        self.threshold = threshold

    def diff(self, a: Arr, b: Arr):
        d = cv2.absdiff(a, b)
        if self.threshold > 0:
            d = cv2.threshold(d, self.threshold, 255, cv2.THRESH_BINARY)[1]
        if self.dilate > 0:
            d = cv2.dilate(d, None, iterations=self.dilate)
        d = d / 255
        d = 1 - d
        score = np.mean(d > 0.1)
        return d, score


class NNDiffer(RMSDiffer):
    def __init__(self, multichannel=False):
        super().__init__(multichannel=multichannel, dilate=-1, threshold=-1)
        self.model: Model = load_model("data/models/test.hdf5", compile=False)

    def compare(self, a: Arr, b: Arr) -> Tuple[Arr, float]:
        diff, _ = super().compare(a, b)
        img = cv2.resize(diff, (IM_WIDTH, IM_HEIGHT), interpolation=cv2.INTER_NEAREST)
        img = np.reshape(img, (1,) + img.shape)
        res = self.model.predict(img)
        return diff, res[0][0]


def colorize_diff(diff: np.ndarray, score: float, th: float):
    # th = 1
    diff = diff.astype(np.float)
    if score > th:
        ns = (score - th) / (1 - th)  # how close to 1.0 score
        diff[:, :, 1] = np.maximum(diff[:, :, 1], 255 * ns)
    else:
        ns = (th - score) / th  # how close to 0.0 score
        m = max(1, int(th / (1 - th)))  # normalize green/red colorization
        diff[:, :, 2] = np.maximum(diff[:, :, 2], 255 * ns * m)
    diff[diff > 255] = 255
    diff = diff.astype(np.uint8)
    return diff


class DiffRegistry:
    def __init__(self):
        self._mapper = self.make_mapper()
        self._index = list(self._mapper.values())

    def make_mapper(self):
        mapper = {
            "ssim": SSIMDiffer(),
            "rms": RMSDiffer(dilate=0),
            "nn": NNDiffer(),
        }
        return mapper

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._index[item]
        return self._mapper[item]

    def __len__(self):
        return len(self._mapper)

    def index(self, item):
        return self._index.index(item)
