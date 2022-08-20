import numpy as np

def numerov(x_start, x_end, N):

    x = np.linspace(x_start, x_end, N)
    print(x)


def main():
    numerov(-10, 10, 100)


if __name__ == "__main__":
    main()