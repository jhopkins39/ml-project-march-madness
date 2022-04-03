import pandas as pd
import numpy as np

class Dataset():
    def __init__(self, datadir="march-ml-mania-dataset/"):
        self.datadir = datadir
        self.regular_results = pd.read_csv(datadir + "RegularSeasonDetailedResults.csv")
        self.tourney_results = pd.read_csv(datadir + "TourneyDetailedResults.csv")

        # Ignoring season and daynum in the headers
        self.compact_headers = pd.read_csv(datadir + "RegularSeasonCompactResults.csv", header=0, nrows=0).columns.tolist()[2:]
        self.detailed_headers = regular_results.columns.tolist()[2:]

        df = pd.read_csv(datadir + "Teams.csv")
        self.teams = dict(zip(df.Team_Id, df.Team_Name))
        self.seasons = pd.read_csv(datadir + "Seasons.csv")

        self.seeds = pd.read_csv(datadir + "TourneySeeds.csv")
        self.slots = pd.read_csv(datadir + "TourneySlots.csv")

    def getTeam(self, id):
        return self.teams[id]

    def getYears(self):
        return self.Seasons.unique()

    def getSeeds(self, season):
        df = self.seeds.loc[seeds['Season'].isin(list(season))]
        return dict(zip(df.Team, df.Seed))
