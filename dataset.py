import pandas as pd
import numpy as np

class Dataset():
    def __init__(self, datadir="march-ml-mania-dataset/"):
        self.datadir = datadir
        self.regular_results = pd.read_csv(datadir + "RegularSeasonDetailedResults.csv")
        self.tourney_results = pd.read_csv(datadir + "TourneyDetailedResults.csv")
        self.compact_headers = pd.read_csv(datadir + "RegularSeasonCompactResults.csv", header=0, nrows=0).columns.tolist()

        self.teams = pd.read_csv(datadir + "Teams.csv", index_col='Team_Id').to_dict()['Team_Name']
        self.seasons = pd.read_csv(datadir + "Seasons.csv")

        self.seeds = pd.read_csv(datadir + "TourneySeeds.csv")
        self.slots = pd.read_csv(datadir + "TourneySlots.csv")

    def getYears(self):
        return self.Seasons.unique()

    def getTeam(self, id):
        return self.teams[id]

    def getSeed(self, season):
        return self.seeds.loc[seeds['Season'].isin(list(season))].set_index('Team').to_dict()['Seed']
