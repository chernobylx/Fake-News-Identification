"""Loading of the ISOT Fake News Dataset.

The dataset ships as two CSV files (``True.csv`` and ``Fake.csv``); this module
combines them into a single labeled DataFrame ready for modeling.
"""
import os
import pandas as pd


class Data:
    """Loads the ISOT dataset and exposes it as features ``X`` and labels ``y``.

    Parameters
    ----------
    path : str
        Directory containing ``True.csv`` and ``Fake.csv``. Defaults to
        ``../data``, i.e. running from the ``notebooks/`` directory.
    """

    def __init__(self, path='../data'):
        self.path = path

    def load(self):
        """Read both CSVs, add a boolean ``Real`` label, and concatenate them.

        Returns ``self`` with attributes ``df`` (full frame), ``X`` (title,
        text, subject, date) and ``y`` (the ``Real`` label).
        """
        self.real = pd.read_csv(os.path.join(self.path, 'True.csv'))
        self.real['Real'] = True

        self.fake = pd.read_csv(os.path.join(self.path, 'Fake.csv'))
        self.fake['Real'] = False

        self.df = pd.concat([self.real, self.fake], ignore_index=True)

        self.y = self.df['Real']
        self.X = self.df.drop(columns=['Real'])
        return self
