import numpy as np


class perceptron:

    def __init__(self):
        self.w = None
        self.x_sample_shape = None

        self.accuracy_lst = None
        self.iter = None
        self.reset_all()

    def iter_reset(self):
        self.w = None
        self.x_sample_shape = None
        self.predict_lst = list()

    def reset_all(self):
        self.w = None
        self.x_sample_shape = None

        self.accuracy_lst = list()
        self.iter = -1
        self.iter_reset()

    def fit(self, X, y):

        check_false_classification = False
        X = np.asarray(X)

        if (len(X.shape) == 1):
            X = np.reshape(X, (1, X.shape[0]))
        if (self.x_sample_shape is None):  # perform init to w (zeros) and dimension d
            self.x_sample_shape = X.shape[1]
            self.w = np.zeros(self.x_sample_shape, dtype=float)
        if (X.shape[0] != len(y)):
            print("Error: x,y size mismatch")
            exit(1)
        while (not check_false_classification):
            finish_loop = True
            for i, x_i in enumerate(X):
                x_i = np.array(x_i, dtype=float)
                y_i = y[i]  ## label is scalar
                try_predict_label = np.dot(self.w, x_i)
                check_prediction = y_i * try_predict_label
                if (check_prediction <= 0):
                    self.w = np.add(self.w, np.dot(y_i, x_i))
                    finish_loop = False
                    break
            if (finish_loop):
                check_false_classification = True

    def predict(self, x):
        self.predict_lst = list()
        x = np.asarray(x, dtype=float)
        if (len(x.shape) == 1):
            x = np.reshape(x, (1, x.shape[0]))
        for sample in x:
            predict_label = np.dot(self.w, sample)
            predict_label = -1 if predict_label <= 0 else 1
            self.predict_lst.append(predict_label)

    def check_accuracy(self, y_label, itr):
        if (self.iter != itr):
            self.iter = itr
            self.accuracy_lst.append(0)
        count = 0
        for i, y_i in enumerate(y_label):
            if (y_i == self.predict_lst[i]):
                count += 1
        self.accuracy_lst.append(self.accuracy_lst.pop() + count)


def main():
    X_samples = [[2.7810836, 2.550537003],
                 [1.465489372, 2.362125076],
                 [3.396561688, 4.400293529],
                 [1.38807019, 1.850220317],
                 [3.06407232, 3.005305973],
                 [7.627531214, 2.759262235],
                 [5.332441248, 2.088626775],
                 [6.922596716, 1.77106367],
                 [8.675418651, -0.242068655],
                 [7.673756466, 3.508563011]]

    # y_labels = [-1, -1, -1, -1, -1, 1, 1, 1, 1,1]
    #
    # perceptron_obj = perceptron()
    # perceptron_obj.fit(X_samples, y_labels)
    # perceptron_obj.predict(X_samples)

# if __name__ == "__main__":
#     main()
