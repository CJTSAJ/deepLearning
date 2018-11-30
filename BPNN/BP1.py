from util import *


class HiddenLayer:
    def __init__(self, inputSize, outputSize):
        self.hiddenCells = np.zeros(outputSize)
        self.weights = np.random.randn(inputSize, outputSize)
        self.hidden_deltas = np.zeros(outputSize)
        self.util = util()
        self.inputCells = np.zeros(inputSize)
        self.inputCorrect = np.zeros((inputSize, outputSize))


    def forward(self, inputs):
        # execute
        self.hiddenCells = np.dot(inputs, self.weights)
        for j in range(len(self.hiddenCells)):
            self.hiddenCells[j] = self.util.logistic(self.hiddenCells[j])
        # output
        return self.hiddenCells[:]

    def backward(self, output_deltas, output_weights):
        # get hidden layer error
        for h in range(len(self.hiddenCells)):
            error = np.dot(output_deltas, output_weights[h])
            self.hidden_deltas[h] = self.util.logistic_derivative(self.hiddenCells[h]) * error

        return self.hidden_deltas[:]

    def update(self, learnRate, correctRate):
        # update input weights
        for i in range(len(self.inputCells)):
            for h in range(len(self.hiddenCells)):
                change = self.hidden_deltas[h] * self.inputCells[i]
                self.weights[i][h] += learnRate * change + correctRate * self.inputCorrect[i][h]
                self.inputCorrect[i][h] = change


class BPNN:
    def __init__(self):
        self.inputCells = []
        self.hiddenLayers = []
        self.outputCells = []
        self.outputWeights = []
        self.outputCorrect = []
        self.util = util()

    #ni: number of input node  nh: number of hidden node  nhl: number of hidden layer
    def setData(self, ni, nh, nhl, no):
        # init cells
        self.inputCells = [1.0] * ni
        self.outputCells = [1.0] * no

        tmpLayers = []
        tmpLayers.append(HiddenLayer(ni, nh))
        for i in range(nhl-1):
            tmpLayers.append(HiddenLayer(nh, nh))
        self.hiddenLayers = tmpLayers

        # random activate
        self.outputWeights = np.random.randn(nh, no)

        # init correctRateion matrix
        self.outputCorrect = np.zeros((nh, no))

    def forward(self, inputs):
        # input layer
        for i in range(len(self.inputCells) - 1):
            self.inputCells[i] = inputs[i]
        # hidden layer
        tmpInput = self.inputCells
        for layer in self.hiddenLayers:
            layer.forward(tmpInput)
            tmpInput = layer.hiddenCells

        # output layer
        self.outputCells = np.dot(tmpInput, self.outputWeights)
        for i in range(len(self.outputCells)):
            self.outputCells[i] = self.util.logistic(self.outputCells[i])
        return self.outputCells[:]

    # correctRate improve the efficiency
    def backward(self, case, label, learnRate, correctRate):
        # get output layer error
        output_deltas = [0.0] * len(self.outputCells)
        for o in range(len(self.outputCells)):
            error = label[o] - self.outputCells[o]
            output_deltas[o] = self.util.logistic_derivative(self.outputCells[o]) * error

        num = len(self.hiddenLayers)
        tmp_output_deltas = output_deltas
        tmp_output_weights = self.outputWeights
        for i in range(num):
            tmp_output_deltas = self.hiddenLayers[num - 1 - i].backward(tmp_output_deltas, tmp_output_weights)
            self.hiddenLayers[num - 1 - i].update(learnRate, correctRate)
            tmp_output_weights = self.hiddenLayers[num - 1 - i].weights
       # update output weights
        lastHiddenCells = self.hiddenLayers[num-1].hiddenCells
        for h in range(len(lastHiddenCells)):
            for o in range(len(self.outputCells)):
                change = output_deltas[o] * lastHiddenCells[h]
                self.outputWeights[h][o] += learnRate * change + correctRate * self.outputCorrect[h][o]
                self.outputCorrect[h][o] = change
        # get global error
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.outputCells[o]) ** 2
        return error

    def train(self, samples, labels, limit, learnRate, correctRate):
        for j in range(limit):
            error = 0.0
            for i in range(len(samples)):
                label = labels[i]
                case = samples[i]
                self.forward(case)
                error += self.backward(case, label, learnRate, correctRate)


'''
        for i in range(len(self.inputCells)):
            for h in range(len(self.hiddenCells)):
                self.inputWeights[i][h] = rand(-0.2, 0.2)

        for h in range(len(self.hiddenCells)):
            for o in range(len(self.outputCells)):
                self.outputWeights[h][o] = rand(-1.0, 1.0)
'''

