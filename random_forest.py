from sklearn.ensemble import RandomForestClassifier
import logistic_regression as lr
from sklearn.metrics import r2_score



# trains and runs a singular random forest model
def use_rf_model(n_estimators=500, max_depth=5, num_components=None):


	data = lr.process_data()

	# np.set_printoptions(threshold=sys.maxsize)
	# print(data)

	X, y, test_X, test_y = lr.separate_data(data)
	if num_components != None:
		X, test_X = lr.apply_PCA(X, test_X, num_components)

	model = RandomForestClassifier(n_estimators=100, max_depth=10, bootstrap=True)
	model.fit(X,y)

	predictions = model.predict(test_X)
	score = model.score(test_X, test_y)

	r2 = r2_score(predictions, test_y)
	print("r2_score: ")
	print(r2)

	print("random forest predictions: ")
	print(predictions)

	print("actual answers: ")
	print(test_y)

	print("random forest score: ")
	print(score)

	return score

use_rf_model()