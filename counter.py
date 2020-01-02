import enum
from time import time

import cv2
import numpy as np

import differ
import shaders
from utils import add_text, get_cap_size, noop, set_cap_size


class Mode(enum.Enum):
    normal = enum.auto()
    ghost = enum.auto()
    record = enum.auto()
    delay = enum.auto()


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
    def __init__(
        self,
        win_name="app",
        delay=33,
        buf_size=2,
        device=0,
        shader="waves",
        threshold=0.9,
    ):
        assert 0 < delay
        assert 1 < buf_size < 100

        self.win_name = win_name
        self.delay = delay
        self.buf_size = buf_size
        self.device = device
        self.shader = shader
        self.threshold = threshold

        self.mode = Mode.ghost
        self.cap = None
        self.base = None
        self.buf = []
        self.time = None
        self.count = 0
        self.moved = False
        self.diff_registry = differ.DiffRegistry()
        self.differ = self.diff_registry[2]

    def init(self, device):
        if device == -1:
            self.cap = MockedDevice(self.shader)
        else:
            self.cap = self.get_camera(device=int(device), scale=0.1)
        self.setup_window()

    def get_camera(self, device=0, scale=1.0):
        cap = cv2.VideoCapture(device)
        w, h = get_cap_size(cap)
        set_cap_size(cap, w * scale, h * scale)
        return cap

    def place_window_on_top(self):
        """SoF said that it works on mac, but it doesn't..."""
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.win_name, np.zeros((100, 100), np.uint8))
        cv2.setWindowProperty(
            self.win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        cv2.waitKey(1)
        cv2.setWindowProperty(self.win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(self.win_name)

    def setup_window(self):
        cv2.namedWindow(self.win_name)
        cv2.createTrackbar(
            "diff type", self.win_name, 0, len(self.diff_registry) - 1, noop
        )
        cv2.setTrackbarPos(
            "diff type", self.win_name, self.diff_registry.index(self.differ)
        )

    def ghost_mode(self, frame):
        self.buf.append(frame)
        if len(self.buf) > self.buf_size:
            self.buf.pop(0)
        if self.buf:
            f, score = self.differ.compare(self.buf[0], self.buf[-1])
            f = differ.colorize_diff(f, score, self.threshold)
            return f

    def record_mode(self, frame):
        f, score = self.differ.compare(self.base, frame)
        f = differ.colorize_diff(f, score, self.threshold)

        if score > self.threshold and self.moved:
            self.moved = False
            self.count += 1

        if score <= self.threshold:
            self.moved = True

        add_text(f, text=f"score: {score:.2f}", pos=(1, 0), anchor=(1, 1))
        add_text(
            f,
            text=f"{self.count}",
            size=6,
            thickness=20,
            pos=(0.5, 0.5),
            anchor="center",
        )
        return f

    def init_record_mode(self, frame):
        self.mode = Mode.record
        self.base = frame.copy()
        self.count = 0
        self.moved = False

    def delay_mode(self, frame):
        target = 5
        e = time() - self.time
        f = frame.copy()
        add_text(
            f,
            f"starting in {target - e:.2f}s",
            size=2,
            pos=(0.5, 0.5),
            anchor="center",
        )
        if e >= target:
            self.init_record_mode(frame)
        return f

    def wait_key(self, frame):
        diff_type = cv2.getTrackbarPos("diff type", self.win_name)
        self.differ = self.diff_registry[diff_type]
        key = cv2.waitKey(self.delay)
        if key == ord("n"):
            self.mode = Mode.normal
        elif key == ord("r"):
            self.init_record_mode(frame)
        elif key == ord("g"):
            self.mode = Mode.ghost
        elif key == ord("d"):
            self.mode = Mode.delay
            self.time = time()
        elif key in (ord("q"), 27):  # esc==27
            return False
        return True

    def _run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)  # vertical flip / mirror

            if not self.wait_key(frame):
                break

            if self.mode == Mode.ghost:
                f = self.ghost_mode(frame)
                if f is None:
                    continue
            elif self.mode == Mode.record:
                f = self.record_mode(frame)
            elif self.mode == Mode.delay:
                f = self.delay_mode(frame)
            else:
                f = frame

            cv2.imshow(self.win_name, f)

    def run(self):
        if self.cap is None:
            self.init(self.device)
        try:
            self._run()
        except KeyboardInterrupt:
            return

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
