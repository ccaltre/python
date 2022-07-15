import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_california_housing

# generate dataset

class Examples:

    @staticmethod
    def ex1():
        X, y = mglearn.datasets.make_forge()
        # plot dataset
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
        plt.legend(["Class 0", "Class 1"], loc=4)
        plt.xlabel("First feature")
        plt.ylabel("Second feature")
        print("X.shape: {}".format(X.shape))
        plt.show()

    @staticmethod
    def ex2():
        X, y = mglearn.datasets.make_wave(n_samples=40)
        plt.plot(X, y, "o")
        plt.xlabel("Feature")
        plt.ylabel("Target")
        plt.show()

    @staticmethod
    def ex3():
        cancer = load_breast_cancer()
        print(cancer.keys())
        print(cancer.data.shape)
        print("Target names: {}".format(cancer.target_names))
        print("Feature names: {}".format(cancer.feature_names))
        return cancer

    @staticmethod
    def ex4():
        carolina = fetch_california_housing()
        print("X shape: {}".format(carolina.data.shape))
        print("Parameters callable from dataset: {}".format(carolina.keys()))
        print("Carolina features: {}".format(carolina.feature_names))
        return carolina


if __name__ == "__main__":
    Examples.ex4()

