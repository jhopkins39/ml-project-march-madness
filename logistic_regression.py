import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import csv
import dataset
import sys
from sklearn.decomposition import PCA
import data_processing


def use_model(num_components=None):

	dp = data_processing.DataProcess()

	data = dp.get_data()


	# np.set_printoptions(threshold=sys.maxsize)
	# print(data)

	X, y, test_X, test_y = dp.get_separated_data()
	if num_components != None:
		X, test_X = apply_PCA(X, test_X, num_components)

	model = LogisticRegression(max_iter=100000)
	model.fit(X, y)

	predictions = model.predict(test_X)
	print("model predictions: ")
	print(predictions)

	train_predictions = model.predict(X)
	print("model train data predictions: ")
	print(train_predictions)

	print("actual: ")
	print(test_y)

	probabilities = model.predict_proba(test_X)
	print("probabilities: ")
	print(probabilities)

	score = model.score(test_X, test_y)
	print("score: ")
	print(score)

	evs = explained_variance_score(predictions, test_y)
	print("explained_variance_score: ")
	print(evs)

	mape = mean_absolute_percentage_error(predictions, test_y)
	print("mean_absolute_percentage_error: ")
	print(mape)

	r2 = r2_score(predictions, test_y)
	print("r2_score: ")
	print(r2)

	dp.analyze_results(probabilities, test_y)



	f1 = f1_score(test_y, predictions)
	print()
	print("f1 score: ", f1)

	recall = recall_score(test_y, predictions)

	print("recall score: ", recall)

	precision = precision_score(test_y, predictions)

	print("precision score: ", precision)

	f1_train = f1_score(y, train_predictions)
	print()
	print("train f1 score: ", f1_train)

	recall_train = recall_score(y, train_predictions)

	print("train recall score: ", recall_train)

	precision_train = precision_score(y, train_predictions)

	print("training precision score: ", precision_train)

	return probabilities, test_y


use_model()









