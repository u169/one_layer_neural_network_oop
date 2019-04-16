import random

random.seed(0)


class Layer:

    def __init__(self, length, activation_function, back_function):
        self.neurons = list()

        self.activation_function = activation_function
        self.back_function = back_function

        for _ in range(length + 1):
            neuron = Neuron(back_function)
            self.neurons.append(neuron)

    def out(self, data: list):
        results = list()

        data.insert(0, 1)

        for neuron_data, neuron in zip(data, self.neurons):
            out = neuron.out(neuron_data)
            results.append(out)

        return self.activation_function(sum(results))

    def retrain(self, data, expected):
        outs = list()

        for i in range(len(data)):
            neuron = self.neurons[i]
            d = data[i]
            out = neuron.out(d)
            outs.append(out)

        out = self.activation_function(sum(outs))
        error = expected - out
        b_error = self.back_function(error)

        for i in range(len(data)):
            neuron = self.neurons[i]
            d = data[i]
            neuron.retrain(b_error)

        error = sum([n._error for n in self.neurons])/len(self.neurons)
        print("Error: {}".format(error))


class Neuron:

    def __init__(self, back_function):
        self.weight = random.random() * 2 - 1

        self.back_function = back_function

        self.__last_in = None
        self.__last_out = None
        self._error = None

    def out(self, data):
        self.__last_in = data
        self.__last_out = self.weight * data
        return self.__last_out

    def retrain(self, b_e_d):
        self._error = b_e_d
        adjustment = self.__last_in * b_e_d
        self.weight += adjustment
