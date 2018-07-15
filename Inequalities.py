import matplotlib.pyplot as plt
import numpy as np


class Inequalities:
    def __init__(self):
        self.data = np.random.binomial(1, 0.25, (100000, 1000))
        self.epsilon_lst = [0.5, 0.25, 0.1, 0.01, 0.001]
        self.plot_opacity = 0.7
        self.p = 0.25
        self.index = np.array([i for i in range(1, self.data.shape[1] + 1)], dtype=int)

    def plot_sessions(self):
        plt.figure(1)
        for i in range(5):
            Xm_lst = list()
            for m in range(self.data.shape[1]):
                Xm = np.sum(self.data[i][:m]) / (m + 1)
                Xm_lst.append(Xm)
            plt.plot(self.index, Xm_lst, alpha=self.plot_opacity, label="Session " + str(i + 1))
        plt.xlabel("m Samples")
        plt.ylabel("Xm Average outcomes of m samples")
        plt.title("5 Session of 1,000 Coin toss Average Probability [EX1 Q23 (a)]")
        plt.legend()
        plt.show()

    def plot_upper_bounds(self):

        calc_chebyshev = lambda m, epsilon: min(1, (1 / (4 * m * (epsilon ** 2))))
        calc_hoeffding = lambda m, epsilon: min(1, 2 * np.exp(-2 * m * (epsilon ** 2)))
        bound_func = [calc_chebyshev, calc_hoeffding]
        for count, epsilon in enumerate(self.epsilon_lst):
            plt.figure(count + 1)
            for itr, bound_iter in enumerate(['Chebyshev', 'Hoeffding']):
                bound_lst = list()
                for m in range(1, self.data.shape[1] + 1):
                    bound = bound_func[itr](m, epsilon)
                    bound_lst.append(bound)
                plt.plot(self.index, bound_lst, alpha=self.plot_opacity, label="Bound " + bound_iter)
            plt.xlabel("m Samples")
            plt.ylabel("Probability")
            plt.title("ε=" + str(epsilon) + " Bounds of 1,000 coin toss [EX1 Q23 (b)]")
            plt.legend()

    def find_number_of_sessions(self):
        accumulate_sessions = np.cumsum(self.data, 1)
        d_mean = accumulate_sessions / self.index
        data_mean_minus_epsilon = np.abs(d_mean - self.p)
        for count, epsilon in enumerate(self.epsilon_lst):
            plt.figure(count + 1)
            inequality_check = (data_mean_minus_epsilon >= epsilon)
            sum_of_good_sessions = np.sum(inequality_check, 0)
            good_session_ratio = sum_of_good_sessions / (self.data.shape[0])
            plt.plot(self.index, good_session_ratio, alpha=self.plot_opacity, label="% Required Set")
            plt.legend(loc=1)
        plt.show()


if __name__ == '__main__':
    inequality_obj = Inequalities()
    inequality_obj.plot_sessions()  # part 1
    inequality_obj.plot_upper_bounds()  # part 2
    inequality_obj.find_number_of_sessions()  # part 3
