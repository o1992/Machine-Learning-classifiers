import random
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def get_data_file():
    file = open('spam.data', 'r')
    df = file.readlines()
    df = [i.strip().split(" ") for i in df]
    file.close()
    return df


def logistic_model(df):
    random.shuffle(df)
    test_data = df[:1000]
    test_label = [i.pop() for i in test_data]
    train_data = df[1000:]
    train_label = [i.pop() for i in train_data]
    test_data = np.array(test_data, dtype=float)
    train_data = np.array(train_data, dtype=float)
    train_label = np.array(train_label, dtype=float)
    l_r_model = LogisticRegression()
    l_r_model.fit(train_data, train_label)

    prob = l_r_model.predict_proba(test_data)
    prob_to_1 = [i[1] for i in prob]
    prob_to_1 = np.array(prob_to_1)
    prob_to_1 = np.argsort(prob_to_1)
    F_P_R = list()
    T_P_R = list()
    N_P = test_label.count('1')
    N_N = 1000 - N_P

    for i in range(N_P):
        correct = 0
        N_i = 0
        while (correct < i):
            correct = (correct + 1) if test_label[prob_to_1[N_i]] == '1' else correct
            N_i += 1
        F_P_R.append(i / N_P)
        T_P_R.append((N_i - i) / N_N)
    F_P_R.append(1)
    T_P_R.append(1)
    plt.title("Linear Logistic Model: ROC of spam.txt")
    plt.xlabel("FPR: False Positive Rate")
    plt.ylabel("TPR: True Positive Rate")
    plt.plot(F_P_R, T_P_R)
    plt.show()


if __name__ == '__main__':
    df = get_data_file()
    logistic_model(df)
