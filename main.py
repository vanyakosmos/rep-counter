from argser import parse_args
from argser.utils import args_to_dict

from counter import RepCounter


class Args:
    delay = 33, "delay between calculations in milliseconds"
    buf_size = 2, "how many frames to save for ghost mode"
    device = -1, "-1 is mock video, >= 0 - real video devices on machine"
    shader = "waves", "placeholder animation for mocked video device"


def main():
    args = parse_args(Args, show=True)
    params = args_to_dict(args)
    counter = RepCounter(**params)
    counter.run()
    counter.release()


if __name__ == "__main__":
    main()
