import os
import pandas as pd
class Data:
    def __init__(self, path = '../data'):
        self.path = path
        pass

    def load(self):
        self.real = pd.read_csv(os.path.join(self.path, 'true.csv'))
        self.real['Real'] = self.real.title.apply(lambda x: True)

        self.fake = pd.read_csv(os.path.join(self.path, 'fake.csv'))
        self.fake['Real'] = self.fake.title.apply(lambda x: False)
        self.df = pd.concat([self.real, self.fake], ignore_index=True)

        self.y = self.df['Real']
        self.X = self.df.drop(columns=['Real'])
        return self



