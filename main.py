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
    retrain_range = 5000

    def activation_function(x): return 1 / (1 + 2.71828**(-x))

    # def back_activation_function(x): return x
    def back_activation_function(x): return (x**2) * [-1, 1][x > 0]
    # def back_activation_function(x): return x * (1 - x)

    layer = Layer(2, activation_function, back_activation_function)

    for _ in range(retrain_range):
        data, expected = input_expected()
        error = layer.train(data, expected)
        print('In: {}\tExpected: {}\tError: {}'.format(data, expected, error))

    print([n.weight for n in layer.neurons])

    print(layer.out([1, 1]))
    print(layer.out([1, 0]))
    print(layer.out([0, 1]))
    print(layer.out([0, 0]))


if __name__ == '__main__':
    main()
