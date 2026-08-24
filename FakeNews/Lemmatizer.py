"""Token-to-string conversion step for the text-processing pipeline."""
from sklearn.base import TransformerMixin
import pandas as pd


class Lemmatizer(TransformerMixin):
    """Converts filtered spaCy tokens back into strings for vectorization.

    Multi-word named entities are joined with underscores (``White_House``)
    so TF-IDF treats them as a single term.
    """

    def fit(self, X, y=None):
        return self

    def process_token(self, token):
        if token.ent_type_:
            return '_'.join(token.text.split())
        return token.text

    def lemmatize_doc(self, doc):
        return [self.process_token(token) for token in doc]

    def transform(self, X, y=None):
        print('Lemmatizing...')
        X = pd.DataFrame(X)
        for col in X.columns:
            X[col] = X[col].apply(self.lemmatize_doc)
            X[col] = X[col].apply(' '.join)
        return X

    def get_params(self, deep=True):
        return {}

    def get_feature_names_out(self, names_in):
        return names_in
