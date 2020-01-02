from typing import Tuple, Union

import cv2
import numpy as np


def get_cap_size(cap):
    return cap.get(3), cap.get(4)


def set_cap_size(cap, width, height):
    cap.set(3, int(width))
    cap.set(4, int(height))


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


def noop(*args, **kwargs):
    pass
