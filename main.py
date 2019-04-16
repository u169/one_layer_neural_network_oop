import random

from src.NnClasses import Layer


def input_expected():
    storage = [
        ([1, 1], 1),
        ([1, 0], 1),
        ([0, 1], 1),
        ([0, 0], 0),
    ]

    return storage[random.randint(0, len(storage) - 1)]


def main():
    retrain_range = 10000

    def a_f(x): return 1 / (1 + 2.71828**(-x))

    # def b_f(x): return x
    def b_f(x): return (x**2) * [-1, 1][x > 0]
    # def b_f(x): return x * (1 - x)

    layer = Layer(2, a_f, b_f)

    for _ in range(retrain_range):
        i, e = input_expected()
        layer.out(i)
        layer.retrain(i, e)

    print([n.weight for n in layer.neurons])

    print(layer.out([1, 1]))
    print(layer.out([1, 0]))
    print(layer.out([0, 1]))
    print(layer.out([0, 0]))


if __name__ == '__main__':
    main()
