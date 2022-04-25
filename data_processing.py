import numpy as np
import csv
import dataset
import sys
from sklearn.decomposition import PCA
import dataset


class DataProcess():

	def __init__(self):

		self.test_train = 1/14

		self.headerNames = self.get_header_names()
		self.final4teams = self.get_final4teams()
		self.team_ids = list()
		self.data = self.process_data()

		self.team_ids = np.array(self.team_ids)
		self.predicting_team_ids = self.team_ids[int(len(self.team_ids)*(1-self.test_train)):]

		self.X, self.y, self.test_X, self.test_y = self.separate_data(self.data)


	def get_data(self):
		return self.data

	def get_separated_data(self):
		return self.X, self.y, self.test_X, self.test_y

	def get_team_ids(self):
		return self.team_ids


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# INTIALIZATION HELPER METHODS

	# 4774 data points - only 56 with a label of 1
	# returns a numpy array where the last onecolumn is the label and the other 30 something columns are the data point averages
	# should average correctly and returns exactly what we want to use these algorithms
	def process_data(self):

		ds = dataset.Dataset()
		years = ds.getYears(compact=False)
		years.remove(2017)
		final4teams = self.final4teams

		final_stats = np.zeros(())
		counter = 0

		team_identifiers = list()

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

				team_identifiers.append((year, team))

		self.team_ids = team_identifiers
		return final_stats

	# gets your labels and your test / train data set
	# the test_train number is exactly one season's worth of data - essentially we'll be guessing 2016 based on 2003-2015 data
	def separate_data(self, data, test_train=(1/14)):

		self.test_train = test_train
		N = len(data)

		train = data[0:int(N*(1-test_train))]
		test = data[int(N*(1-test_train)):]

		d = len(train[0])

		train_labels = train[:,-1]
		test_labels = test[:,-1]

		train = train[:,0:-1]
		test = test[:,0:-1]

		train[np.isnan(train)] = np.median(train[~np.isnan(train)])
		train_labels[np.isnan(train_labels)] = np.median(train_labels[~np.isnan(train_labels)])
		test[np.isnan(test)] = np.median(test[~np.isnan(test)])
		test_labels[np.isnan(test_labels)] = np.median(test_labels[~np.isnan(test_labels)])

		return train, train_labels, test, test_labels


	# the header names of the original dataframe from dataset class
	def get_header_names(self):
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
	def get_final4teams(self):
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

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	# runs principle componenet analysis on my trainging and testing data :)
	def apply_PCA(self, num_components):
		# at the current state of the data 4/10/22 4 is the optimal number of components
		pca = PCA(num_components)
		pca.fit(self.X)

		# print("explained_variance_ratio: ")
		# print(pca.explained_variance_ratio_)
		# print("singular_values: ")
		# print(pca.singular_values_)
		# print()
		# print()

		new_train = pca.transform(self.X)
		new_test = pca.transform(self.test_x)
		return new_train, new_test

	# two hyperparameters to "tune" for our project
	# based on playing around with it, PCA does not help and top 4 or top 8 probabilities doesn't matter
	def analyze_results(self, probabilities, actual_y, num_max=8):

		maximum_index = np.argpartition(probabilities[:,1], -num_max)[-num_max:]
		max_probabilities = probabilities[maximum_index]
		actual_from_probabilities = actual_y[maximum_index]
		better_score = np.sum(actual_from_probabilities) / len(actual_from_probabilities)

		ds = dataset.Dataset()
		curr_teams = self.predicting_team_ids[maximum_index]
		curr_names = list()
		for tid in curr_teams[:,1]:
			curr_names.append(ds.getTeam(tid))

		# index_predict = np.argpartition(actual_y, -4)[-4:]
		# answers = actual_y[index_predict]

		print()
		print("best probabilities:  ")
		print(max_probabilities)

		print()
		print("best probabilities indexes:  ")
		print(maximum_index)

		print()
		print("actual values of highest probability likelihood from current model: ")
		print(actual_from_probabilities)

		print()
		print("a better score: ")
		print(better_score)

		print()
		print("year and team id: ")
		print(curr_teams)

		print()
		print("team names: ")
		print(curr_names)

		# ----------------------------------------------------------------------------------------------------------------------
		"""
		Some Notes from Jake
		What our model spits out right now, is that of the eight teams it said had the highest probility of being a final four team
		two of them actually did make the final four.  Furthermore, they were the two teams in the finals.  Of the other 6 teams, 
		One did not make the tournament, but based on visual inspection others had higher seeds as well which is exciting in my opinion

		"""
		# --------------------------------------------------------------------------------------------------------------------------------------------



