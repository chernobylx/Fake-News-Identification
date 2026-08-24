"""spaCy tokenization step for the text-processing pipeline."""
import spacy
import pandas as pd
from sklearn.base import TransformerMixin


class Tokenizer(TransformerMixin):
    """Runs each text column through spaCy's ``en_core_web_sm`` pipeline.

    The parser is disabled for speed and ``merge_entities`` is added so that
    multi-word named entities (e.g. "White House") survive as single tokens
    for the downstream :class:`~FakeNews.Lemmatizer.Lemmatizer`.
    """

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=['parser'])
        self.nlp.add_pipe('merge_entities')

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('Tokenizing...')
        X = pd.DataFrame(X)
        for col in X.columns:
            pipe = self.nlp.pipe(X[col])
            X[col] = [doc for doc in pipe]

        return X.to_numpy()

    def get_params(self, deep=True):
        return {}

    def get_feature_names_out(self, input_features):
        return input_features
