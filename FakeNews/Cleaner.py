class Cleaner(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.X = X
