import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import csv
import dataset

import seaborn as sns



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

# print(final4teams)
# print(final4teams[2007])

def process_data():

	ds = dataset.Dataset()
	years = ds.getYears(compact=False)
	years.remove(2017)
	
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

def get_test_data():
	return np.array([
		 ['W01', '0.0'],
		 ['W02', '0.0'],
		 ['W03', '0.0'],
		 ['W04', '0.0'],
		 ['W05', '0.0'],
		 ['W06', '0.0'],
		 ['W07', '0.0'],
		 ['W08', '0.0'],
		 ['W09', '0.0'],
		 ['W10', '0.0'],
		 ['W11a', '1.0'],
		 ['W11b', '0.0'],
		 ['W12', '0.0'],
		 ['W13', '0.0'],
		 ['W14' ,'0.0'],
		 ['W15' ,'0.0'],
		 ['W16a', '0.0'],
		 ['W16b' ,'0.0'],
		 ['X01' ,'0.0'],
 		 ['X02' ,'0.0'],
		 ['X03' ,'1.0'],
		 ['X04' ,'0.0'],
		 ['X05' ,'0.0'],
		 ['X06' ,'0.0'],
		 ['X07' ,'0.0'],
		 ['X08' ,'0.0'],
		 ['X09' ,'0.0'],
		 ['X10' ,'0.0'],
		 ['X11' ,'0.0'],
		 ['X12' ,'0.0'],
		 ['X13' ,'0.0'],
		 ['X14' ,'0.0'],
		 ['X15' ,'0.0'],
		 ['X16' ,'0.0'],
		 ['Y01' ,'1.0'],
		 ['Y02' ,'0.0'],
		 ['Y03' ,'0.0'],
		 ['Y04' ,'0.0'],
		 ['Y05' ,'0.0'],
		 ['Y06' ,'0.0'],
		 ['Y07' ,'0.0'],
		 ['Y08' ,'0.0'],
		 ['Y09' ,'0.0'],
		 ['Y10' ,'0.0'],
		 ['Y11a' ,'0.0'],
		 ['Y11b' ,'0.0'],
		 ['Y12' ,'0.0'],
		 ['Y13' ,'0.0'],
		 ['Y14' ,'0.0'],
		 ['Y15' ,'0.0'],
		 ['Y16' ,'0.0'],
		 ['Z01' ,'1.0'],
		 ['Z02' ,'0.0'],
		 ['Z03' ,'0.0'],
		 ['Z04' ,'0.0'],
		 ['Z05' ,'0.0'],
		 ['Z06' ,'0.0'],
		 ['Z07' ,'0.0'],
		 ['Z08' ,'0.0'],
		 ['Z09' ,'0.0'],
		 ['Z10' ,'0.0'],
		 ['Z11' ,'0.0'],
		 ['Z12' ,'0.0'],
		 ['Z13' ,'0.0'],
		 ['Z14' ,'0.0'],
		 ['Z15' ,'0.0'],
		 ['Z16a' ,'0.0'],
		 ['Z16b' ,'0.0']])

def get_correct_num(item):
	item = item[1:]
	if item == "11a" or item =='11b' or item == '16a' or item == '16b' or item == '12a' or item == '12b' or item == '14a' or item == '14b' or item == '13a' or item == '13b':
		item = item[0:len(item)-1]
	return int(item)


def use_model():
	data = process_data()
	for i in range(len(data)):
		data[i,0] = get_correct_num(data[i,0])
	test = get_test_data()

	print()
	print()
	print()
	for i in range(len(test)):
		test[i,0] = get_correct_num(test[i,0])


	clf = LogisticRegression(random_state=0)

	train = data[:,0].reshape(len(data),1)
	testing = test[:,0].reshape(len(test),1)

	clf.fit(train, data[:,1])

	predictions = clf.predict(testing)
	print("model predictions: ")
	print(predictions)

	probabilities = clf.predict_proba(testing)
	print("probabilities: ")
	print(probabilities)

	score = clf.score(testing, test[:,1])
	print("score: ")
	print(score)

	evs = explained_variance_score(predictions, test[:,1])
	print("explained_variance_score: ")
	print(evs)

	mape = mean_absolute_percentage_error(predictions, test[:,1])
	print("mean_absolute_percentage_error: ")
	print(mape)

	r2 = r2_score(predictions, test[:,1])
	print("r2_score: ")
	print(r2)


	sns.regplot(x=train, y=data[:,1], data=data, logistic=True, ci=None)




	# plt.scatter(data[:,0], data[:,1])
	# plt.show()
		


use_model()
