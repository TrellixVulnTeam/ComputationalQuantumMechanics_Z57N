import numpy as np

def numerov(N, x_start, x_end):

    x = np.linspace(x_start, x_end, N)
    print(x)


def main():
    numerov(0.3, -10, 10)


if __name__ == "__main__":
    main()