from BP import BPNN
import pandas as pd
import numpy as np
def readfile(file):
    iris_data = pd.read_csv(file);
    iris_data = np.array(iris_data)
    datas = iris_data[:, :len(iris_data[0])-1]
    labels = iris_data[:, len(iris_data[0]) - 1]
    #labels_ = np.zeros((len(labels), 3));
    for i in range(len(labels)):
        if labels[i] == "Iris-setosa":
            labels[i] = 0.1
        elif labels[i] == "Iris-versicolor":
            labels[i] = 0.2
        elif labels[i] == "Iris-virginica":
            labels[i] = 0.3
    labels = np.array(labels).reshape(len(iris_data), 1)
    #print(labels)
    return datas, labels
if __name__ == '__main__':
    readfile("iris.txt")
    samples, labels = readfile("iris.txt")
    nn = BPNN()
    nn.setData(len(samples[0]), 5, len(labels[0]))
    nn.train(samples, labels, 500, 0.05, 0.3)

    samples1, labels1 = readfile("test.txt")
    for case in samples1:
        print(nn.forward(case))
    '''nn = BPNN()
    samples = [
        [0.05, 0.1],
        [2, 3]
    ]
    labels = [[0.01,0.99], [0.2, 0.3]]
    nn.setData(2, 8, 1, 2)
    nn.train(samples, labels, 1000, 0.05, 0.3)
    for case in samples:
        print(nn.forward(case))'''
