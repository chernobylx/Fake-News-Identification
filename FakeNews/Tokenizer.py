import spacy
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
class Tokenizer(TransformerMixin):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=['parser'])
        self.nlp.add_pipe('merge_entities')

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X)
        cols = X.columns
        for col in cols:
            pipe = self.nlp.pipe(X[col])
            X[col] = [doc for doc in pipe]

        return X.to_numpy()
    def get_params(self, deep=True):
        return {}

    def get_feature_names_out(self, input_features):
        return input_features