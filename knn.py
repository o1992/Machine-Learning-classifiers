import random
import numpy as np
import matplotlib.pyplot as plt
import copy


class knn:
    def __init__(self, k):
        self.k = k
        self.X = []
        self.y = []

    def fit(self, X, y):
        self.X = copy.deepcopy(X)
        self.y = copy.deepcopy(y)

    def predict(self, x_predict):
        dist = list()
        for i in self.X:
            dist.append(np.linalg.norm(x_predict - i))
        dist = np.argsort(dist)

        same_type = 0
        for i in range(self.k):
            same_type = same_type + 1 if self.y[dist[i]] == '0' else same_type
        return 0 if (same_type >= int(self.k / 2)) else 1


def main():
    df = get_data_file()
    random.shuffle(df)
    test_data = df[:1000]
    test_label = [i.pop() for i in test_data]
    train_data = df[1000:]
    train_label = [i.pop() for i in train_data]
    test_data = np.array(test_data, dtype=float)
    train_data = np.array(train_data, dtype=float)
    train_label = np.array(train_label)
    k_lst = [1, 2, 5, 10, 100]
    error = list()
    for k in k_lst:
        print(k)
        knn_model = knn(k)
        knn_model.fit(train_data, train_label)
        counter = 0
        for i in range(1000):
            counter = counter + 1 if (knn_model.predict(test_data[i]) != (int(test_label[i]))) else counter
        error.append(counter / 1000)
    plot_result(k_lst, error)


def get_data_file():
    file = open('spam.data', 'r')
    df = file.readlines()
    df = [i.strip().split(" ") for i in df]
    file.close()
    return df


def plot_result(k_lst, error_lst):
    plt.title("KNN Classifier: Error rate of k nearest match")
    plt.xlabel("k nearest match")
    plt.ylabel("Error Rate")
    plt.plot(k_lst, error_lst)
    plt.show()


if __name__ == "__main__":
    main()
