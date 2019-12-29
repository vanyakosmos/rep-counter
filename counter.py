import enum
from time import time
from typing import Tuple, Union

import cv2
import numpy as np

from skimage.metrics import structural_similarity

import shaders


def get_cap_size(cap):
    return cap.get(3), cap.get(4)


def set_cap_size(cap, width, height):
    cap.set(3, int(width))
    cap.set(4, int(height))


def calc_diff(base, frame):
    base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score, diff = structural_similarity(base, frame, full=True)
    diff = (diff + 1) / 2
    diff = (diff * 255).astype(np.uint8)
    return cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR), score


def calculate_origin(im_size, text_size, pos, anchor):
    w, h = im_size
    tw, th = text_size
    if isinstance(anchor, str):
        if anchor == "center":
            anchor = (0.5, 0.5)
        elif anchor == "left":
            anchor = (1.0, 1.0)
    pw, ph = pos
    pw, ph = w * pw, h * ph
    aw, ah = anchor
    pw -= tw * aw
    ph += th * ah
    return int(pw), int(ph)


def add_text(
    img: np.ndarray,
    text: str,
    size=1,
    color=(255, 0, 0),
    thickness=1,
    pos=(0, 0),
    anchor: Union[str, Tuple[float, float]] = (0, 0),
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w, *_ = img.shape
    text_size, _ = cv2.getTextSize(text, font, size, thickness)
    org = calculate_origin((w, h), text_size, pos, anchor)
    cv2.putText(img, text, org, font, size, color, thickness, cv2.LINE_AA)


def rot_append(arr: list, el, size=5):
    arr.append(el)
    if len(arr) > size:
        arr.pop(0)


def noop(*args, **kwargs):
    pass


class Mode(enum.Enum):
    normal = enum.auto()
    ghost = enum.auto()
    record = enum.auto()


class MockedDevice:
    def __init__(self, shader):
        self.start = time()
        self.shader = getattr(shaders, shader)

    def isOpened(self):
        return True

    def read(self):
        t = time() - self.start
        h, w, d = (480, 640, 3)
        img = self.shader(h, w, t)
        return True, shaders.norm_img(img)

    def release(self):
        pass


class RepCounter:
    def __init__(self, win_name="app", delay=33, buf_size=2, device=0, shader="waves"):
        assert 0 < delay
        assert 1 < buf_size < 100
        self.win_name = win_name
        self.delay = delay
        self.buf_size = buf_size
        self.device = device
        self.shader = shader
        self.mode = Mode.normal
        self.cap = None
        self.base = None
        self.buf = []

    def init(self):
        if self.device == -1:
            self.cap = MockedDevice(self.shader)
        else:
            self.cap = self.get_camera(device=int(self.device), scale=0.1)
        self.setup_window()

    def get_camera(self, device=0, scale=1.0):
        cap = cv2.VideoCapture(device)
        w, h = get_cap_size(cap)
        set_cap_size(cap, w * scale, h * scale)
        return cap

    def place_window_on_top(self):
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.win_name, np.zeros((100, 100), np.uint8))
        cv2.setWindowProperty(
            self.win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        cv2.waitKey(1)
        cv2.setWindowProperty(self.win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(self.win_name)

    def setup_window(self):
        self.place_window_on_top()
        cv2.namedWindow(self.win_name)
        cv2.createTrackbar("mode", self.win_name, 0, 2, noop)
        cv2.createTrackbar("diff type", self.win_name, 0, 2, noop)

    def ghost_mode(self, frame):
        rot_append(self.buf, frame, size=self.buf_size)
        if self.buf:
            f, _ = calc_diff(self.buf[0], self.buf[-1])
            return f

    def record_mode(self, frame):
        f, score = calc_diff(self.base, frame)
        add_text(f, text=f"score: {score:.2f}", pos=(1, 0), anchor=(1, 1))
        add_text(f, text=f"start", size=4, thickness=2, pos=(0.5, 0.5), anchor="center")
        return f

    def wait_key(self, frame):
        key = cv2.waitKey(self.delay)
        if key == ord("n"):
            self.mode = Mode.normal
        elif key == ord("r"):
            self.mode = Mode.record
            self.base = frame.copy()
        elif key == ord("g"):
            self.mode = Mode.ghost
        elif key in (ord("q"), 27):  # esc==27
            return False
        return True

    def _run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)  # vertical flip / mirror

            if self.mode == Mode.ghost:
                f = self.ghost_mode(frame)
                if f is None:
                    continue
            elif self.mode == Mode.record:
                f = self.record_mode(frame)
            else:
                f = frame

            cv2.imshow(self.win_name, f)

            if not self.wait_key(frame):
                break

    def run(self):
        if self.cap is None:
            self.init()
        try:
            self._run()
        except KeyboardInterrupt:
            return

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
