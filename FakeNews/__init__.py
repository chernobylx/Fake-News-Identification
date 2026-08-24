"""Custom scikit-learn transformers for fake news classification.

The modules in this package implement the data loading, cleaning, and
NLP preprocessing steps used by the pipelines in ``notebooks/report.ipynb``:

- :class:`FakeNews.Data.Data` — loads the ISOT dataset
- :class:`FakeNews.Cleaner.Cleaner` — drops duplicates, junk rows, and stubs
- :class:`FakeNews.Tokenizer.Tokenizer` — spaCy tokenization with entity merging
- :class:`FakeNews.Filter.Filter` — removes punctuation, stop words, whitespace
- :class:`FakeNews.Lemmatizer.Lemmatizer` — tokens back to vectorizable strings
- :class:`FakeNews.Predictor.Predictor` — maps K-Means clusters to class labels
"""
