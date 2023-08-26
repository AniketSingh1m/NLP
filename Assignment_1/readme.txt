# Amazon Fine Food Reviews - Text Classification Assignment

This repository contains code and information related to the assignment involving text classification of Amazon Fine Food Reviews. The goal of the assignment is to predict the "Score" (ratings between 1 to 5) based on the text reviews using various preprocessing steps, vectorization techniques, and machine learning algorithms.

## Table of Contents

- [Dataset](#dataset)
- [Preprocessing Steps](#preprocessing-steps)
- [Vectorization](#vectorization)
- [Machine Learning Algorithms](#machine-learning-algorithms)
- [Results and Comparison](#results-and-comparison)

## Dataset

The dataset used for this assignment can be downloaded from [this Kaggle link](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews). It contains Amazon Fine Food Reviews including text reviews and corresponding ratings.

## Preprocessing Steps

The text data underwent the following preprocessing steps using the NLTK library:

- Tokenization: Splitting the text into individual words or tokens.
- Lemmatization: Reducing words to their base or dictionary form.
- Data Cleansing: Removing stopwords, symbols, and URLs from the text.

## Vectorization

The text data was vectorized using the following techniques:

- CountVectorizer: Converting text into a matrix of token counts.
- TFIDFVectorizer: Converting text into a matrix of TF-IDF features.
- Word2Vec: Creating word embeddings using the Word2Vec algorithm.
- GoogleNews Word2Vec: Utilizing pre-trained Word2Vec embeddings from Google News.

## Machine Learning Algorithms

The following machine learning algorithms were applied to the vectorized data:

- Logistic Regression
- Support Vector Classifier (SVC)
- Random Forest

Each algorithm was evaluated using the various vectorization techniques.

## Results and Comparison

The results of the assignment were compared using the Classification Report generated for each combination of algorithm and vectorization technique. The following metrics were analyzed:

- Precision
- Recall
- F1-score
- Accuracy

For detailed code and analysis, please refer to the provided [Jupyter Notebook](link_to_your_notebook.ipynb) and the [PDF Report](link_to_your_pdf_report.pdf).


