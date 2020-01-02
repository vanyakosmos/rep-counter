from time import time

import cv2

from differ import RMSDiffer
from utils import get_cap_size, set_cap_size
from .model import IM_HEIGHT, IM_WIDTH


DELAY = 100


def main():
    differ = RMSDiffer(multichannel=True, dilate=-1, threshold=-1)
    cap = cv2.VideoCapture(0)
    print(get_cap_size(cap))
    set_cap_size(cap, IM_WIDTH, IM_HEIGHT)

    base = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        key = cv2.waitKey(DELAY)
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            base = frame.copy()

        if base is None:
            cv2.imshow("recorder", frame)
        else:
            diff, _ = differ.compare(base, frame)
            diff = cv2.resize(
                diff, (IM_WIDTH, IM_HEIGHT), interpolation=cv2.INTER_NEAREST
            )
            cv2.imwrite(f"data/train/{time():.2f}.png", diff)
            cv2.imshow("recorder", diff)


if __name__ == "__main__":
    main()
