from itertools import permutations
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import accuracy_score
class Predictor(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, W, y):
        '''
        W: m_articles x n_categories numpy array from svc
        y: m_articles x 1 numpy array of labels
        returns: self
        '''
        self.W = W
        self.yt = y

        #the predicted class is the column with the largest value for each row
        self.preds = self.W.argmax(axis = 1)
        self.labels = y.unique()

        #generate permutations of class labels
        self.perms = permutations(self.labels)

        #initialize best permutation
        self.best = (0, self.labels)

        #score each permutation and update best
        for perm in self.perms:
            yp = [perm[pred] for pred in self.preds]
            score = accuracy_score(self.yt, yp)
            if score > self.best[0]:
                self.best = (score, perm)

        #save best permutation
        self.label_mapping = self.best[1]

        return self

    def predict(self, W):
        yp = W.argmax(axis = 1)
        yp = [self.label_mapping[y] for y in yp]
        return yp

    def score(self, W, yt):
        yp = self.predict(W)
        return accuracy_score(yt, yp)

    def fit_predict(self, W, y):
        self.fit(W, y)
        return self.predict(W)