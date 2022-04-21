import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import csv
import dataset
import sys
from sklearn.decomposition import PCA

# the header names of the original dataframe from dataset class
def get_header_names():
	headerNames = [
		'Wscore',
		'Lscore',
		'Wloc',  
		'Numot',

		'Wfgm',  
		'Wfga',
		'Wfgm3',  
		'Wfga3',  
		'Wftm',  
		'Wfta',   
		'Wor',   
		'Wdr',  
		'Wast',   
		'Wto',  
		'Wstl',  
		'Wblk',   
		'Wpf',  

		'Lfgm',  
		'Lfga',  
		'Lfgm3',  
		'Lfga3',  
		'Lftm',  
		'Lfta',   
		'Lor',   
		'Ldr',  
		'Last',   
		'Lto',  
		'Lstl',  
		'Lblk',   
		'Lpf'
		]
	return headerNames


# dictionary of all the final four teams for each year - easily indexed
def get_final4teams():
	final4teams = {
		2003: (1242,1266,1393,1400),
		2004: (1163, 1181,1210, 1329),
		2005: (1228, 1257, 1314, 1277),
		2006: (1196	,1206, 1417, 1261),
		2007: (1196, 1417, 1326, 1207),
		2008: (1242, 1314, 1272, 1417),
		2009: (1277, 1163, 1314, 1437),
		2010: (1139, 1277, 1181, 1452),
		2011: (1139, 1433, 1163, 1246),
		2012: (1242, 1326, 1246, 1257),
		2013: (1257, 1455, 1276, 1393),
		2014: (1163, 1196, 1246, 1458),
		2015: (1181, 1277, 1458, 1246),
		2016: (1314, 1393, 1437, 1328)
	}
	return final4teams


# 4774 data points - only 56 with a label of 1
# returns a numpy array where the last onecolumn is the label and the other 30 something columns are the data point averages
# should average correctly and returns exactly what we want to use these algorithms
def process_data():

	ds = dataset.Dataset()
	years = ds.getYears(compact=False)
	years.remove(2017)
	final4teams = get_final4teams()

	final_stats = np.zeros(())
	counter = 0

	for year in years:
		game_teams, game_stats = ds.getRegularGames(season = year, compact=False)
		game_teams = np.array(game_teams)
		game_stats = np.array(game_stats)
		unique_teams = list(set(game_teams[:,0]))

		for team in unique_teams:
			team_indices_w = np.where(game_teams[:,0] == team)
			team_indices_l = np.where(game_teams[:,1] == team)
			info_to_average_w = game_stats[team_indices_w]
			info_to_average_l = game_stats[team_indices_l]
			num_w = len(info_to_average_w)
			num_l = len(info_to_average_l)
			w_mean = np.mean(info_to_average_w, axis=0)[4:17]
			l_mean = np.mean(info_to_average_l, axis=0)[17:]
			w_score = np.mean(info_to_average_w, axis=0)[0]
			l_score = np.mean(info_to_average_l, axis=0)[1]
			label = 1 if team in final4teams[year] else 0
			curr_final = [num_w, num_l, w_score, l_score, *list(w_mean), *list(l_mean), label]
			if counter == 0:
				final_stats = np.array(curr_final)
				counter += 1
			else:
				final_stats = np.vstack((final_stats,curr_final))

	return final_stats

# gets your labels and your test / train data set
# the test_train number is exactly one season's worth of data - essentially we'll be guessing 2016 based on 2003-2015 data
def separate_data(data, test_train=(1/14)):

	N = len(data)

	train = data[0:int(N*(1-test_train))]
	test = data[int(N*(1-test_train)):]

	d = len(train[0])

	train_labels = train[:,d-1]
	test_labels = test[:,d-1]

	train = train[:,0:d-2]
	test = test[:,0:d-2]

	train[np.isnan(train)] = np.median(train[~np.isnan(train)])
	train_labels[np.isnan(train_labels)] = np.median(train_labels[~np.isnan(train_labels)])
	test[np.isnan(test)] = np.median(test[~np.isnan(test)])
	test_labels[np.isnan(test_labels)] = np.median(test_labels[~np.isnan(test_labels)])

	return train, train_labels, test, test_labels

# runs principle componenet analysis on my trainging and testing data :)
def apply_PCA(train_x, test_x, num_components):
	# at the current state of the data 4/10/22 4 is the optimal number of components
	pca = PCA(num_components)
	pca.fit(train_x)

	# print("explained_variance_ratio: ")
	# print(pca.explained_variance_ratio_)
	# print("singular_values: ")
	# print(pca.singular_values_)
	# print()
	# print()

	new_train = pca.transform(train_x)
	new_test = pca.transform(test_x)
	return new_train, new_test


def use_model(num_components=None):

	data = process_data()

	# np.set_printoptions(threshold=sys.maxsize)
	# print(data)

	X, y, test_X, test_y = separate_data(data)
	if num_components != None:
		X, test_X = apply_PCA(X, test_X, num_components)

	model = LogisticRegression(max_iter=100000)
	model.fit(X, y)

	predictions = model.predict(test_X)
	print("model predictions: ")
	print(predictions)

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

	return probabilities, test_y

# two hyperparameters to "tune" for our project
# based on playing around with it, PCA does not help and top 4 or top 8 probabilities doesn't matter
def analyze_results(num_components=None, num_max=8):

	probabilities, actual_y = use_model(num_components)

	maximum_index = np.argpartition(probabilities[:,1], -num_max)[-num_max:]
	max_probabilities = probabilities[maximum_index]
	actual_from_probabilities = actual_y[maximum_index]
	better_score = np.sum(actual_from_probabilities) / len(actual_from_probabilities)

	# index_predict = np.argpartition(actual_y, -4)[-4:]
	# answers = actual_y[index_predict]

	print()
	print("best probabilities:  ")
	print(max_probabilities)

	print()
	print("best probabilities indexes:  ")
	print(maximum_index)

	print()
	print("actual values of highest probability likelihood from logistic regression model: ")
	print(actual_from_probabilities)

	print()
	print("a better score: ")
	print(better_score)

#analyze_results()









