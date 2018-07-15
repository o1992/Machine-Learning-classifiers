# Lets importing some libraries
import numpy as np  # linear algebra
import pandas as pd  # for manipulating datasets
import seaborn as sb
from pylab import rcParams
import matplotlib.pyplot as plt


class PricePrediction:
    def __init__(self, data_file_name):
        self.df = pd.read_csv(data_file_name)
        self.err = []




    def clean_data(self):
        """ pre-process first stage, clear corrupted data"""
        self.df.dropna(inplace=True)
        to_remove_list = []
        for index, row in self.df.iterrows():
            if row['price'] < 0:
                to_remove_list.append(index)
            elif row['sqft_lot15'] < 0:
                to_remove_list.append(index)
        print(to_remove_list)
        self.df.drop(to_remove_list, axis=0, inplace=True)

    def find_correlate_data(self):
        """ pre-process second stage, find and keep the correlated data and remove the uncorrelated data"""
        rcParams['figure.figsize'] = 5, 4
        sb.set_style('whitegrid')
        # Lets see the some important stats
        f, ax = plt.subplots(figsize=(17, 14))
        sb.heatmap(self.df.corr(), annot=True, annot_kws={'size': 8}, linewidths=.5, fmt='.2f', ax=ax)
        plt.show()

    def data_remove_uncorrelated(self):
        """ pre-process stage 3, remove the data which uncorrelated to the price."""

        # remove date and id columns, not relevant
        self.remove_cols = ["date", "id", "long", "lat"]
        # remove uncorrelate data as we analyze the heat-map
        self.df.drop(self.remove_cols, axis=1, inplace=True)
        # noise vector
        noise = [1 for i in range(len(self.df))]
        self.df.insert(loc=0, column="Noise", value=noise)
        self.df = pd.get_dummies(self.df, columns=['zipcode'])


    def linear_regression(self, slice_precentage):
        """ Linear regression - learning of the matrix, and using the weighted matrix onto a new data slice
        to check the error of the prediction"""

        # create learning_data and testing_data
        learn_df = self.df.sample(frac=(slice_precentage / 100), random_state=slice_precentage + 11)
        self.df.sample()
        learning_indexes = learn_df.index
        test_df = self.df.drop(learning_indexes)
        # create the y vectors
        y_learn = learn_df['price']
        y_test = test_df['price']
        # remove 'price' col
        learn_df.drop(['price'], 1, inplace=True)
        test_df.drop(['price'], 1, inplace=True)
        self.calc_USV_T(learn_df)
        self.calc_w_curl(y_learn)
        # check learned error and test error
        # get the results and calculate the error
        learn_matrix = np.dot(learn_df, self.w_curl)
        err_learned = self.error_calc(learn_matrix, y_learn)
        result_matrix = np.dot(test_df, self.w_curl)
        err_test = self.error_calc(result_matrix, y_test)
        self.err.append((slice_precentage, err_learned, err_test))

    def calc_USV_T(self, X):  # X†
        X = X.as_matrix()
        x_t = np.transpose(X)
        self.USV_T = np.linalg.pinv(x_t)  # U * SIGMA_DAGGER * V_t

    def calc_w_curl(self, Y):  # ŵ
        X_D_T = np.transpose(self.USV_T)
        self.w_curl = np.dot(X_D_T, Y)

    def error_calc(self, learn, expected):
        err = np.array((learn - expected) ** 2).mean()
        return err

    def plot_error(self):
        index, learn_err, test_err = zip(*self.err)
        plt.plot(index, learn_err, label="Learned data-set Error")
        plt.plot(index, test_err, label="New data-set Error")
        plt.xlabel("% of data allocated for learning")
        plt.ylabel("Error")
        plt.title("House Price Prediction - Error Comparison")
        plt.legend()
        plt.ylim(0, 5 * (10 ** 10), 1)
        remove_cols_desc = "removed features:" + str(self.remove_cols)
        plt.text(0.95, 0.06, remove_cols_desc,
                 verticalalignment='bottom', horizontalalignment='left',

                 color='green', fontsize=9)
        plt.show()


def iter_linear_regression():
    for i in range(1, 100):
        if ((i + 1) % 10 == 0):
            print(str(i + 1) + '%', end=" ", flush=True)
        if (i == 1):
            print(str(0) + '%', end=" ", flush=True)

        price_prediction_obj.linear_regression(i)


if __name__ == '__main__':
    price_prediction_obj = PricePrediction('kc_house_data.csv')

    price_prediction_obj.clean_data()
    price_prediction_obj.find_correlate_data()

    price_prediction_obj.data_remove_uncorrelated()
    iter_linear_regression()
    price_prediction_obj.plot_error()
