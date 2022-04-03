import pandas as pd

class Dataset():
    def __init__(self, datadir="march-ml-mania-dataset/"):
        self.datadir = datadir

        regular_comp = pd.read_csv(datadir + "RegularSeasonCompactResults.csv")
        tourney_comp = pd.read_csv(datadir + "TourneyCompactResults.csv")
        regular_det = pd.read_csv(datadir + "RegularSeasonDetailedResults.csv")
        tourney_det = pd.read_csv(datadir + "TourneyDetailedResults.csv")

        self.regular_results = pd.concat((regular_comp.loc[regular_comp['Season'] < 2003], regular_det))
        self.tourney_results = pd.concat((tourney_comp.loc[tourney_comp['Season'] < 2003], tourney_det))

        # Ignoring season and daynum in the headers
        self.compact_headers = regular_comp.columns.tolist()[2:]
        self.detailed_headers = regular_det.columns.tolist()[2:]

        df = pd.read_csv(datadir + "Teams.csv")
        self.teams = dict(zip(df.Team_Id, df.Team_Name))
        self.seasons = pd.read_csv(datadir + "Seasons.csv")

        self.seeds = pd.read_csv(datadir + "TourneySeeds.csv")
        self.slots = pd.read_csv(datadir + "TourneySlots.csv")

    def getTeam(self, id):
        return self.teams[id]

    def getYears(self):
        return self.seasons.Season.unique().tolist()

    def getSeeds(self, season):
        df = self.seeds.loc[self.seeds['Season'] == season]
        return dict(zip(df.Team, df.Seed))

    def getRegularGames(self, season=None, compact=True):
        if type(season) is int: season = [season]
        headers = self.compact_headers if compact else self.detailed_headers
        if season is None:
            return self.regular_results[headers]
        return self.regular_results.loc[self.regular_results['Season'].isin(list(season))][headers]

    def getTourneyGames(self, season=None, compact=True):
        if type(season) is int: season = [season]
        headers = self.compact_headers if compact else self.detailed_headers
        if season is None:
            return self.tourney_results[headers]
        return self.tourney_results.loc[self.tourney_results['Season'].isin(list(season))][headers]

