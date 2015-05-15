import pandas as pd
import numpy as np
from sklearn import ensemble, feature_extraction, preprocessing, cross_validation, metrics

def validate():
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, labels, test_size=0.2)
	# Train a random forest classifier
	# 0.5428 with (500, 25, 2)
	n_estimators = 500
	max_features = 25
	min_samples_split = 2
	for max_features in [10,15,20,25,30]:
		rf_clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators, max_features = max_features, min_samples_split = min_samples_split, verbose=1)
		rf_clf.fit(X_train, y_train)
		rf_preds = rf_clf.predict_proba(X_test)
		print metrics.log_loss(y_test, rf_preds)

	# # Train a gradient boosting classifier
	# n_estimators = 50
	# max_depth = 6
	# for max_depth in [6,7,8,9,10]:
	# 	gbm_clf = ensemble.GradientBoostingClassifier(n_estimators=n_estimators, max_depth = max_depth, verbose=1)
	# 	gbm_clf.fit(X_train, y_train)
	# 	gbm_preds = gbm_clf.predict_proba(X_test)
	# 	print metrics.log_loss(y_test, gbm_preds)

	# preds = (rf_preds+gbm_preds)/2
	# print metrics.log_loss(y_test, preds)


def predict():
	# Train a random forest classifier
	rf_clf = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=1000, max_features = 25, min_samples_split=2, verbose=1)
	rf_clf.fit(train, labels)
	rf_preds = rf_clf.predict_proba(test)

	# Train a gradient boosting classifier
	gbm_clf = ensemble.GradientBoostingClassifier(n_estimators=60, max_depth = 6, verbose=1)
	gbm_clf.fit(train, labels)
	gbm_preds = gbm_clf.predict_proba(test)

	preds = (rf_preds*0.8+gbm_preds*0.2)
	# create submission file
	preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
	preds.to_csv('benchmark.csv', index_label='id')

if __name__ == '__main__':
	# import data
	train = pd.read_csv('../data/train.csv')
	test = pd.read_csv('../data/test.csv')
	sample = pd.read_csv('../data/sampleSubmission.csv')

	# drop ids and get labels
	labels = train.target.values
	train = train.drop('id', axis=1)
	train = train.drop('target', axis=1)
	test = test.drop('id', axis=1)

	# encode labels 
	lbl_enc = preprocessing.LabelEncoder()
	labels = lbl_enc.fit_transform(labels)

	predict()
