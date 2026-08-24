"""Token filtering step for the text-processing pipeline."""
from sklearn.base import TransformerMixin
import pandas as pd


class Filter(TransformerMixin):
    """Drops punctuation, stop words, and whitespace tokens from spaCy docs."""

    def filter(self, token):
        return not (token.is_punct or token.is_stop or token.is_space)

    def process_doc(self, doc):
        return [token for token in doc if self.filter(token)]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('Filtering...')
        X = pd.DataFrame(X)
        for col in X.columns:
            X[col] = X[col].apply(self.process_doc)

        return X.to_numpy()

    def get_params(self, deep=True):
        return {}

    def get_feature_names_out(self, features_in):
        return features_in
