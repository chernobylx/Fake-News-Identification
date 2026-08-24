"""Cluster-to-label assignment for unsupervised pipelines.

K-Means (and NMF) produce arbitrary cluster/component indices with no notion
of which one means "real". This estimator learns the mapping from cluster
index to class label by scoring every possible permutation against the
training labels and keeping the best one.
"""
from itertools import permutations
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import accuracy_score
import pandas as pd


class Predictor(TransformerMixin, BaseEstimator):
    """Maps cluster assignments to class labels via best-permutation search."""

    def fit(self, W, y):
        """Learn the cluster-to-label mapping.

        Parameters
        ----------
        W : ndarray of shape (n_articles, n_clusters)
            Cluster affinity matrix; each row's argmax is its assigned cluster.
        y : array-like of shape (n_articles,)
            True class labels.
        """
        self.W = W
        self.yt = pd.Series(y)

        # the predicted cluster is the column with the largest value per row
        self.preds = self.W.argmax(axis=1)
        self.labels = pd.Series(y).unique()

        # score each permutation of label assignments and keep the best
        self.best = (0, self.labels)
        for perm in permutations(self.labels):
            yp = [perm[pred] for pred in self.preds]
            score = accuracy_score(self.yt, yp)
            if score > self.best[0]:
                self.best = (score, perm)

        self.label_mapping = self.best[1]
        return self

    def predict(self, W):
        print('Predicting...')
        yp = W.argmax(axis=1)
        return [self.label_mapping[y] for y in yp]

    def score(self, W, yt):
        return accuracy_score(yt, self.predict(W))

    def fit_predict(self, W, y):
        self.fit(W, y)
        return self.predict(W)
