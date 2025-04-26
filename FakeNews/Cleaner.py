from sklearn.base import TransformerMixin
import pandas as pd
class Cleaner(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.X = X
        return self

    def transform(self, df, y=None):
        df = df.drop(df[df.duplicated(subset = 'text')].index)
        df = df.drop(df[df.duplicated(subset = 'title')].index)

        df = df.drop(df[df.date.str.contains('http')].index)
        df = df.drop(df[df.date.str.contains('MSN')].index)

        df.date = pd.to_datetime(df.date, format = 'mixed')

        df['word_count_text'] = df.text.str.split().apply(len)
        df['word_count_title'] = df.title.str.split().apply(len)


        df.drop(df[df.word_count_text < df.word_count_title].index, inplace=True)

        return df[['text','title']].to_numpy()

    def get_params(self, deep=True):
        return {}

    def get_feature_names_out(self, features_in):
        return features_in

