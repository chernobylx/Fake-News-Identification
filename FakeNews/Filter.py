from sklearn.base import TransformerMixin
import pandas as pd
class Filter(TransformerMixin):
    def __init__(self):
        pass
    def filter(self, token):
        if token.is_punct:
            return False
        elif token.is_stop:
            return False
        elif token.is_space:
            return False
        else:
            return True
    def get_feauture_names_out(self, features_in):
        return features_in
    def get_feature_names(self):
        return ['a','b','c','d','e','f']

    def process_doc(self, doc):
        return [token for token in doc if self.filter(token)]
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            X[col] = X[col].apply(self.process_doc)

        return X.to_numpy()

    def get_params(self, deep=True):
        return {}

