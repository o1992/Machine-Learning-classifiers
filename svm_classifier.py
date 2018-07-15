from sklearn import svm
import numpy as np
import random
import perceptron
import matplotlib.pyplot as plt


class svm_classifier:
    def __init__(self):
        self.all_points = list()
        self.D2_label = list()
        self.D1_label = list()
        self.D1_2dGauss = None
        self.distribution = None
        self.dim = 2
        self.num_samples = None
        self.prediction = None
        self.accuracy_lst = list()
        self.iter = -1

    def train(self, X, y):

        if (np.asarray(X).shape[0] != np.asarray(y).shape[0]):
            print("err")
        self.clf = svm.SVC(C=1e10, kernel='linear')
        self.clf.fit(X, y)

    def predict(self, X):

        self.prediction = self.clf.predict(X)

    def check_accuracy(self, y, iter):
        if (self.iter != iter):
            self.iter = iter
            self.accuracy_lst.append(0)
        count = 0
        for i, y_i in enumerate(y):
            if (y_i == self.prediction[i]):
                count += 1
        self.accuracy_lst.append(self.accuracy_lst.pop() + count)

    def reset(self):
        self.iter = -1
        self.accuracy_lst = list()

    def do_labeling(self, num_samples, D_num, test):
        self.num_samples = num_samples
        if (D_num == 1):
            while (self.build_D1_points() == False or test):
                if (test):
                    break
                continue
        else:
            while (self.build_D2_points() == False):
                if (test):
                    break
                continue
            self.build_D2_labels()

    def build_D2_labels(self):
        for i in range(len(self.all_points)):
            self.D2_label.append(np.sign(self.all_points[i][1]))

    def build_D1_points(self):
        i2 = [[1, 0],
              [0, 1]]

        self.D1_2dGauss = np.random.multivariate_normal([0, 0], i2, size=self.num_samples)

        return self.build_f_true_classifier()

    def build_f_true_classifier(self):
        count_same_label = 0
        self.D1_label = list()
        w = np.array([0.3, -0.5])

        for i in range(self.num_samples):
            # tmp_label = (np.dot(self.D1_2dGauss[i], w))
            tmp_label = np.sign(np.inner(w, self.D1_2dGauss[i]))
            count_same_label += tmp_label
            self.D1_label.append(tmp_label)
        if (abs(count_same_label) == self.num_samples):
            return False
        else:
            return True

    def build_D2_points(self):

        points1 = list()
        points2 = list()
        self.all_points = list()
        self.D2_label = list()

        for i in range(self.num_samples):

            if (np.random.randint(2) == 1):
                point = (random.uniform(-1, 3), random.uniform(-3, -1))
                points1.append(point)
                self.all_points.append(point)
            # self.D2_label.append(rect1_label)

            else:
                point = (random.uniform(-3, 1), random.uniform(1, 3))
                points2.append(point)
                self.all_points.append(point)
                # self.D2_label.append(rect2_label)
        if (len(points1) == 0 or len(points2) == 0):
            return False
        else:
            return True


def main():
    run_iteration = 500
    k = 10000
    m_samples = [5, 10, 15, 25, 70]
    svm_obj = svm_classifier()
    perceptron_obj = perceptron.perceptron()

    for itr, m in enumerate(m_samples):

        for i in range(run_iteration):
            print(1, i, m)
            svm_obj.do_labeling(m, 1, False)
            svm_obj.train(svm_obj.D1_2dGauss, svm_obj.D1_label)
            perceptron_obj.fit(svm_obj.D1_2dGauss, svm_obj.D1_label)

            svm_obj.do_labeling(k, 1, True)
            svm_obj.predict(svm_obj.D1_2dGauss)
            perceptron_obj.predict(svm_obj.D1_2dGauss)
            perceptron_obj.check_accuracy(svm_obj.D1_label, itr)
            svm_obj.check_accuracy(svm_obj.D1_label, itr)
            perceptron_obj.iter_reset()

    perceptron_obj.accuracy_lst = [i / (run_iteration * k) for i in perceptron_obj.accuracy_lst]
    svm_obj.accuracy_lst = [i / (run_iteration * k) for i in svm_obj.accuracy_lst]
    plt.title("D1 distribution")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of samples (m)")
    plt.plot(m_samples, svm_obj.accuracy_lst, label="SVM Mean accuracy")
    plt.plot(m_samples, perceptron_obj.accuracy_lst, label="Perceptron Mean accuracy")
    plt.legend(loc=4)

    svm_obj.reset()
    perceptron_obj.reset_all()

    for itr, m in enumerate(m_samples):

        for i in range(run_iteration):
            print(2, i, m)
            svm_obj.do_labeling(m, 2, False)
            svm_obj.train(svm_obj.all_points, svm_obj.D2_label)
            perceptron_obj.fit(svm_obj.all_points, svm_obj.D2_label)

            svm_obj.do_labeling(k, 2, True)
            svm_obj.predict(svm_obj.all_points)
            perceptron_obj.predict(svm_obj.all_points)
            perceptron_obj.check_accuracy(svm_obj.D2_label, itr)

            svm_obj.check_accuracy(svm_obj.D2_label, itr)
            perceptron_obj.iter_reset()

    svm_obj.accuracy_lst = [i / (run_iteration * k) for i in svm_obj.accuracy_lst]
    perceptron_obj.accuracy_lst = [i / (run_iteration * k) for i in perceptron_obj.accuracy_lst]
    plt.figure(2)
    plt.title("D2 distribution")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of samples (m)")
    plt.plot(m_samples, svm_obj.accuracy_lst, label="SVM Mean accuracy")
    plt.plot(m_samples, perceptron_obj.accuracy_lst, label="Perceptron Mean accuracy")
    plt.legend(loc=4)
    plt.show()


main()
