import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import pandas as pd
import numpy as np
import csv as csv
from nltk.tokenize import word_tokenize

# Load the train file into a dataframe
print "Loading Data..."
train_df = pd.read_table('../data/train.tsv', header=0)
movie_data = []
for index, row in train_df.iterrows():
	if row['Sentiment'] < 2:
		tup = (row['Phrase'] ,'neg')
	elif row['Sentiment'] > 2:
		tup = (row['Phrase'] ,'pos')
	movie_data.append(tup)

# Tokenize reviews for use by NaiveBayesClassifier
print "Tokenizing Data (Using Sparsed Forms)..."
train_data = [({word: 'true' for word in word_tokenize(x[0])}, x[1]) for x in movie_data]

# Classify Data using NaiveBayesClassifier
print "Start Training on Data..."
classifier = NaiveBayesClassifier.train(train_data)
classifier.show_most_informative_features(n=20)

# Test accuracy of model
# print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
