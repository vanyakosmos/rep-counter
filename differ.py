from typing import Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity

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
        score, diff = structural_similarity(
            a, b, full=True, multichannel=self.multichannel
        )
        diff = (diff + 1) / 2
        return diff, score


class RMSDiffer(Differ):
    def diff(self, a: Arr, b: Arr):
        d = cv2.absdiff(a, b)
        d = cv2.threshold(d, 25, 255, cv2.THRESH_BINARY)[1]
        d = cv2.dilate(d, None, iterations=3)
        d = d / 255
        d = 1 - d
        score = np.mean(d > 0.1)
        return d, score


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
            "rms": RMSDiffer(),
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
