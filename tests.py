import cv2

from counter import RepCounter


def test_record_mode_diff():
    counter = RepCounter()
    counter.base = cv2.imread("assets/base.png")
    frame = cv2.imread("assets/next.png")
    img = counter.record_mode(frame)
    cv2.imwrite("assets/diff.png", img)


def main():
    test_record_mode_diff()


if __name__ == "__main__":
    main()
