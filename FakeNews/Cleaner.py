from sklearn.base import TransformerMixin
import pandas as pd
class Cleaner(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.X = X.copy()
        self.to_drop = {}
        self.to_drop['dup_text'] = self.X[self.X.duplicated(subset = 'text')].index
        self.X.drop(self.to_drop['dup_text'], inplace = True)

        self.to_drop['dup_title'] = self.X[self.X.duplicated(subset = 'title')].index
        self.X.drop(self.to_drop['dup_title'], inplace = True)

        self.to_drop['date_http'] = self.X[self.X.date.str.contains('http')].index
        self.X.drop(self.to_drop['date_http'], inplace = True)

        self.to_drop['date_MSN'] = self.X[self.X.date.str.contains('MSN')].index
        self.X.drop(self.to_drop['date_MSN'], inplace = True)


        self.X['word_count_title'] = self.X.title.str.split().apply(len)
        self.X['word_count_text'] = self.X.text.str.split().apply(len)

        self.to_drop['stubs'] = self.X[self.X.word_count_text < self.X.word_count_title].index
        self.X.drop(self.to_drop['stubs'], inplace = True)
        return self

    def transform(self, X, y=None):
        #print(self.to_drop)
        df = X.copy()
        for index in self.to_drop.values():
           df.drop(index, inplace = True)

        return df.to_numpy()

    def get_params(self, deep=True):
        return {}

    def get_feature_names_out(self, features_in):
        return features_in

