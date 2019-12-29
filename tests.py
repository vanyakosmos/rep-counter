import cv2

from main import RepCounter


def main():
    counter = RepCounter()
    counter.base = cv2.imread("assets/base.png")
    frame = cv2.imread("assets/next.png")
    img = counter.record_mode(frame)
    cv2.imwrite("assets/result.png", img)


if __name__ == "__main__":
    main()
