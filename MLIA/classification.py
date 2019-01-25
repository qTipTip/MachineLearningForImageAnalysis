import numpy as np


class Classifier(object):

    def train(self, X, y):
        """
        Train the classifier.

        :param X: numpy array of shape (m, d) containing m samples with d features each.
        :param y: numpy array of shape (m) containing the m training labels.
        :return:
        """

        raise NotImplementedError('Classifier.train is not implemented yet')

    def predict(self, X, **kwargs):
        """
        Predict labels for test data using this classifier.
        """

        raise NotImplementedError('Classifier.predict is not implemented yet')


class KNN(Classifier):

    def train(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, k=1):
        """
        Predict labels for test data with a KNN-classifier.
        :param X: numpy array of shape (M, d) containing M samples with d features each.
        :param k: number of nearest neighbors
        :return: numpy array of shape (M, ) containing predicted labels for test data.
        """

        distances = self.compute_distances(X)

        return self._predict_labels(distances, k=k)

    def _predict_labels(self, distances, k):
        pass

    def compute_distances(self, X):
        """
        Computes the distance between each test point in X and each training point in self.train_X

        :param X: numpy array of shape (M, d) containing M samples with d features each.
        :return: numpy array of shape (M, m) containing the pairwise distances between each training and test-sample.
        """

        return np.sqrt(
            np.sum(self.train_X ** 2, axis=1) - \
            2 * np.dot(X, self.train_X.T) + \
            np.sum(X ** 2, axis=1)[:, None]
        )