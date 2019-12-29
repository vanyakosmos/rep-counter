import cv2
import numpy as np
from skimage.metrics import structural_similarity


WIN_NAME = 'stream'
DELAY = 1  # in ms


def set_cap_size(cap, width: int, height: int):
    cap.set(3, width)
    cap.set(4, height)


def calc_diff(base, frame):
    base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score, diff = structural_similarity(base, frame, full=True)
    diff = (diff + 1) / 2
    diff = (diff * 255).astype(np.uint8)
    return cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR), score


def add_text(img, text: str, size=4, color=(255, 0, 0), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize, _ = cv2.getTextSize(text, font, size, thickness)
    textX = (img.shape[1] - textsize[0]) // 2
    textY = (img.shape[0] + textsize[1]) // 2
    pos = (textX, textY)
    cv2.putText(img, text, pos, font, size, color, thickness, cv2.LINE_AA)


def test_calc_diff():
    base = cv2.imread('assets/base.png')
    frame = cv2.imread('assets/next.png')
    img, score = calc_diff(base, frame)
    add_text(img, text=f'{score:.2f}')
    cv2.imwrite('assets/result.png', img)


def rot_append(arr: list, el, size=5):
    arr.append(el)
    if len(arr) > size:
        arr.pop(0)


def process():
    print('connecting to camera...')
    cap = cv2.VideoCapture(0)
    set_cap_size(cap, 640, 360)
    # cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print('processing...')
    base = None
    mode = 'ghost'
    buf = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        if mode == 'ghost':
            rot_append(buf, frame, size=2)
            if buf:
                f, _ = calc_diff(buf[0], buf[-1])
            else:
                f = frame
        elif mode == 'record':
            f, score = calc_diff(base, frame)
            add_text(f, text=f'{score:.2f}')
        else:
            f = frame

        cv2.imshow(WIN_NAME, f)
        key = cv2.waitKey(DELAY)
        if key == ord('r'):
            mode = 'record'
            base = frame.copy()
        elif key == ord('g'):
            mode = 'ghost'
        elif key in (ord('q'), 27):  # esc==27
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    # test_calc_diff()
    process()


if __name__ == '__main__':
    main()
