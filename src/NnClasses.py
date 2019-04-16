import random

random.seed(0)


class Layer:

    def __init__(self, length, activation_function, back_activation_function):
        self.neurons = list()

        self.activation_function = activation_function
        self.back_activation_function = back_activation_function

        for _ in range(length + 1):                         # "+ 1" <- bias
            neuron = Neuron(back_activation_function)
            self.neurons.append(neuron)

    def out(self, data: list):
        data_with_bias = [1] + data

        return self.activation_function(
            sum(
                [neuron.out(d)
                 for neuron, d
                 in zip(self.neurons, data_with_bias)]
            ))

    def train(self, data, expected):
        out = self.out(data)

        out_error = self.back_activation_function(expected - out)

        for neuron in self.neurons:
            neuron.train(out_error)

        error = sum([n._last_error for n in self.neurons])/len(self.neurons)
        return error


class Neuron:

    def __init__(self, back_activation_function):
        self.weight = random.random() * 2 - 1

        self.back_activation_function = back_activation_function

        self.__last_in = None
        self.__last_out = None
        self._last_error = None

    def out(self, data):
        self.__last_in = data
        self.__last_out = self.weight * data
        return self.__last_out

    def train(self, error):
        self._last_error = error
        adjustment = self.__last_in * error
        self.weight += adjustment
