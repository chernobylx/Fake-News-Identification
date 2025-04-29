# Fake-News-Identification
Unsupervised Machine Learning Final Project

For this porject I trained 4 pipelines to detect fake news articles using the [ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/ISOT_Fake_News_Dataset_ReadMe.pdf)
The best performing model was a support vector classifier that utilized Term Frequency - Inverse Document Frequency and Principal Component Analysis to achieve a validation accuracy of 99.6%
K-means clustering also performed well, achieving an accuracy of 96.4%.
The trained models are available to load as .pkl files in the models directory. 

The FakeNews directory is a python package that contains several custom Sklearn Transformer classes used to construct the pipelines.
