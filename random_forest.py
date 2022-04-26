from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
import numpy as np
import data_processing
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


# trains and runs a singular random forest model
def use_rf_model(n_estimators=500, max_depth=10, num_components=None):


	dp = data_processing.DataProcess()

	data = dp.get_data()


	# np.set_printoptions(threshold=sys.maxsize)
	# print(data)

	X, y, test_X, test_y = dp.get_separated_data()
	if num_components != None:
		X, test_X = apply_PCA(X, test_X, num_components)


	model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=True)
	model.fit(X,y)

	predictions = model.predict(test_X)
	score = model.score(test_X, test_y)

	r2 = r2_score(predictions, test_y)
	print("r2_score: ")
	print(r2)

	print("random forest predictions: ")
	print(predictions)

	train_predictions = model.predict(X)
	print("model train data predictions: ")
	print(train_predictions)

	print("actual answers: ")
	print(test_y)

	probabilities = model.predict_proba(test_X)
	print("probabilities: ")
	print(probabilities)

	print("random forest score: ")
	print(score)

	dp.analyze_results(probabilities, test_y,4)


	f1 = f1_score(test_y, predictions)
	print()
	print("f1 score: ", f1)

	recall = recall_score(test_y, predictions)

	print("recall score: ", recall)

	precision = precision_score(test_y, predictions,zero_division=1)

	print("precision score: ", precision)

	f1_train = f1_score(y, train_predictions)
	print()
	print("train f1 score: ", f1_train)

	recall_train = recall_score(y, train_predictions)

	print("train recall score: ", recall_train)

	precision_train = precision_score(y, train_predictions)

	print("training precision score: ", precision_train)

	return probabilities, test_y

use_rf_model(n_estimators=100, max_depth=5)
