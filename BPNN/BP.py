from util import *

class BPNN:
    def __init__(self):
        self.inputCells = []
        self.hiddenCells = []
        self.outputCells = []
        self.inputWeights = []
        self.outputWeights = []
        self.inputCorrect = []
        self.outputCorrect = []
        self.util = util()
    def setData(self, ni, nh, no):
        # init cells
        self.inputCells = [1.0] * (ni + 1)
        self.hiddenCells = [1.0] * nh
        self.outputCells = [1.0] * no
        # init weights
        self.inputWeights = np.zeros((ni+1, nh))
        self.outputWeights = np.zeros((nh, no))
        # random activate
        self.inputWeights = np.random.randn(ni+1, nh)
        self.outputWeights = np.random.randn(nh, no)

        # init correctRateion matrix
        self.inputCorrect = np.zeros((ni+1, nh))
        self.outputCorrect = np.zeros((nh, no))

    def forward(self, inputs):
        # input layer
        for i in range(len(self.inputCells) - 1):
            self.inputCells[i] = inputs[i]
        # hidden layer
        self.hiddenCells = np.dot(self.inputCells, self.inputWeights)
        for j in range(len(self.hiddenCells)):
            self.hiddenCells[j] = self.util.logistic(self.hiddenCells[j])
        # output layer
        self.outputCells = np.dot(self.hiddenCells, self.outputWeights)
        for i in range(len(self.outputCells)):
            self.outputCells[i] = self.util.logistic(self.outputCells[i])
        return self.outputCells[:]

    #correctRate improve the efficiency
    def backward(self, case, label, learnRate, correctRate):
        # get output layer error
        output_deltas = [0.0] * len(self.outputCells)
        for o in range(len(self.outputCells)):
            error = label[o] - self.outputCells[o]
            output_deltas[o] = self.util.logistic_derivative(self.outputCells[o]) * error
        # get hidden layer error
        hidden_deltas = [0.0] * len(self.hiddenCells)
        for h in range(len(self.hiddenCells)):
            error = np.dot(output_deltas, self.outputWeights[h])
            hidden_deltas[h] = self.util.logistic_derivative(self.hiddenCells[h]) * error
        # update output weights
        for h in range(len(self.hiddenCells)):
            for o in range(len(self.outputCells)):
                change = output_deltas[o] * self.hiddenCells[h]
                self.outputWeights[h][o] += learnRate * change + correctRate * self.outputCorrect[h][o]
                self.outputCorrect[h][o] = change
        # update input weights
        for i in range(len(self.inputCells)):
            for h in range(len(self.hiddenCells)):
                change = hidden_deltas[h] * self.inputCells[i]
                self.inputWeights[i][h] += learnRate * change + correctRate * self.inputCorrect[i][h]
                self.inputCorrect[i][h] = change
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

