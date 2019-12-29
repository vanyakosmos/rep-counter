from typing import Tuple

import numpy as np


COLOR_BGR2GRAY: int
COLOR_GRAY2BGR: int

WINDOW_NORMAL: int
WINDOW_AUTOSIZE: int
WINDOW_OPENGL: int

FONT_HERSHEY_SIMPLEX: int
LINE_AA: int


class Device:
    def get(self, key: int) -> int: ...

    def set(self, key: int, value: int): ...

    def read(self) -> Tuple[bool, np.ndarray]: ...

    def release(self): ...

    def isOpened(self) -> bool: ...


def VideoCapture(device=0) -> Device: ...


# image processing


def flip(img: np.ndarray, option: int) -> np.ndarray: ...


def cvtColor(frame, mode: int): ...


# GUI

def imread(file: str) -> np.ndarray: ...


def imwrite(file: str, arr: np.ndarray): ...


def imshow(title, color): ...


def waitKey(delay=0): ...


def namedWindow(winname: str, flags=0): ...


def resizeWindow(winname: str, dim: Tuple[int, int]): ...


def destroyAllWindows(): ...


def getTextSize(text: str, font: int, size: int, thickness: int): pass


def putText(img: np.ndarray, text: str, pos: Tuple[int, int], font: int,
            size: int, color: Tuple[int, int, int], thickness: int,
            line_type: int): pass
