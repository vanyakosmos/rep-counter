import numpy as np
from numpy import abs, arctan, cos, log, pi, sin, sqrt
from skimage import io


def get_pos(h, w):
    y = np.linspace(np.zeros(w), np.ones(w), h)
    x = np.linspace(np.linspace(0, 1, w), np.linspace(0, 1, w), h)
    z = np.ones((h, w))
    return y, x, z


def merge(a, b, c):
    return np.stack([a, b, c], axis=2)


def split(img):
    return img[:, :, 0], img[:, :, 1], img[:, :, 2]


def norm_img(img: np.ndarray):
    img = (img - img.min()) / (img.max() - img.min())
    img *= 255
    img = img.astype(np.uint8)
    return img


def pstat(*images: np.ndarray, label="", full=False):
    for img in images:
        info = (
            f"shape {img.shape}, min {img.min()}, max {img.max()}, sample {img[1, 1]}"
        )
        if label:
            info = f"{label}\n{info}"
        print(info)
        if full:
            print(img)
        print("- " * 60)


def waves(h, w, t):
    y, x, z = get_pos(h, w)
    pos = merge(y, x, z)
    pos = pos * 2 - 1
    y, x, z = split(pos)
    mov0 = x + y + cos(sin(t) * 2) * 100 + sin(x / 100) * 1000
    mov1 = y / 0.9 + t
    mov2 = x / 0.2
    c1 = abs(sin(mov1 + t) / 2 + mov2 / 2 - mov1 - mov2 + t)
    c2 = abs(sin(c1 + sin(mov0 / 1000 + t) + sin(y / 40 + t) + sin((x + y) / 100) * 3))
    c3 = abs(sin(c2 + cos(mov1 + mov2 + c2) + cos(mov2) + sin(x / 1000)))
    return merge(c1 / 4.2, c2, c3)


def spiral(h, w, t):
    y, x, z = get_pos(h, w)
    cx = x - 0.5
    cy = y - 0.5
    x = log(sqrt(cx * cx + cy * cy))
    y = arctan(cy, cy)
    hor = 10
    ver = 10
    diag = 10
    arms = 6
    lines = 5
    rings = 5
    spiral_angle = pi / 3
    color = np.zeros_like(cx)
    # color += cos(ver * cy + t)  # ver
    color += cos(hor * cx - t)  # hor
    color += cos(
        2 * diag * (cx * sin(spiral_angle) + cy * cos(spiral_angle)) + t
    )  # diag
    # color += cos(lines * x + t)  # arms
    # color += cos(rings * x - t)  # rings
    color += cos(
        2 * arms * (x * sin(spiral_angle) + y * cos(spiral_angle)) + t
    )  # spiral
    return merge(sin(color + t / 3) * 0.75, color, sin(color + t / 3) * 7.5)


def ripple(h, w, t):
    speed = 0.035
    y, x, z = get_pos(h, w)
    inv_ar = h / w
    col = merge(y, x, z * (0.5 + 0.5 * sin(t)))
    y = (0.5 - y) * inv_ar
    x = 0.5 - x
    r = -sqrt(x * x + y * y)
    z = 0.5 + 0.5 * sin((r * t * speed) / 0.013)
    return col * merge(z, z, z)


def main():
    img = None
    for i in range(10):
        img = spiral(480, 640, i)
        try:
            assert 0 <= img.min() <= img.max() <= 1
        except AssertionError:
            print("ERROR", i)
            pstat(img)
    if img is not None:
        img = norm_img(img)
        io.imsave("assets/shader.png", img)


if __name__ == "__main__":
    main()
