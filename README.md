# Fake-News-Identification
Unsupervised Machine Learning Final Project

For this project I trained 4 sklearn pipelines to detect fake news articles using the [ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/ISOT_Fake_News_Dataset_ReadMe.pdf)
The best performing model was a support vector classifier that utilized Term Frequency - Inverse Document Frequency and Principal Component Analysis to achieve a validation accuracy of 99.6%
K-means clustering also performed well, achieving an accuracy of 96.4%.
The trained models are available to load as .pkl files in the `\models` directory. 

The data used to train the models are in the Fake.csv and True.csv files in the `\data` directory. The data is also available for download from [kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

The `\FakeNews` directory is a python package that contains several custom Sklearn Transformer classes used to construct the pipelines.

The `conda` environment I used is available in the `\env` directory as `project.yml`

