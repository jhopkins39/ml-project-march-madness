import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import csv
import dataset

headernNames = [
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


# right now it adds all the final four values for the teams, paired with their seeds only
def process_data():

	ds = dataset.Dataset()
	years = ds.getYears(compact=False)
	years.remove(2017)

	final4teams = get_final4teams()


	# game_teams, game_stats = ds.getRegularGames(season = years, compact=False)
	# game_teams = np.array(game_teams)
	# game_stats = np.array(game_stats)
	# print(game_teams)
	# print(game_stats)


	final_stats = np.zeros(())

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
			print(w_score)
			print(l_score)
			print(w_mean)
			print(l_mean)
			curr_final = [num_w, num_l, w_score, l_score, *list(w_mean), *list(l_mean)]
			print()
			print(curr_final)
			return curr_final


	









	teamIDs = list()
	stats = list()

	final = list()
	for year in years:
		seeds = ds.getSeeds(season=year)
		
		team = list(seeds.keys())
		seed = np.array(list(seeds.values()))
		
		final4 = final4teams[year]
		curr_indices = list()
		for ffid in final4:
			curr_indices.append(team.index(ffid))

		labels = np.zeros((len(seed)))

		for index in curr_indices:
			labels[index] = 1

		curr_final = np.vstack((seed,labels))
		curr_final = curr_final.T

		for i in range(len(curr_final)):
			final.append(curr_final[i])

	final = np.array(final)
	return final

print(len(process_data()))














