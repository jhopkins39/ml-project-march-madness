"""For loading and reading data files

  Typical usage example:
      from dataset import Dataset
      ds = Dataset()
      ds.<method_name>()
"""
import pandas as pd

class Dataset():
    """Parser for the march-ml-mania dataset.
    
    This class reads the various .csvs provided and provides methods 
    for accessing the game data.

    Attributes:
        datadir: filepath for data directory
        regular_results: game data for the regular season from 1985-2017
            Detailed game data only available for the 2013 season and later (NaNs pre 2013)
        tourney_results: tourney data from 1985-2017
            Detailed game data only available for the 2013 season and later (NaNs pre 2013)
        compact_headers: column names of compact data
        detailed_headers: column names of detailed data
        teams: dictionary mapping team id to team name
        seasons: season and region data
        seeds: seeds of each team for every season
        slots: tourney bracket layout
    """

    def __init__(self, datadir="march-ml-mania-dataset/"):
        """Initializes dataset attributes."""
        self.datadir = datadir

        regular_comp = pd.read_csv(datadir + "RegularSeasonCompactResults.csv")
        tourney_comp = pd.read_csv(datadir + "TourneyCompactResults.csv")
        regular_det = pd.read_csv(datadir + "RegularSeasonDetailedResults.csv")
        tourney_det = pd.read_csv(datadir + "TourneyDetailedResults.csv")

        self.regular_results = pd.concat((regular_comp.loc[regular_comp['Season'] < 2003], regular_det)).replace(['H', 'A', 'N'], [1, -1, 0])
        self.tourney_results = pd.concat((tourney_comp.loc[tourney_comp['Season'] < 2003], tourney_det)).replace(['H', 'A', 'N'], [1, -1, 0])

        self.team_headers = ['Wteam', 'Lteam']
        self.compact_headers = regular_comp.columns.tolist()[2:]
        self.detailed_headers = regular_det.columns.tolist()[2:]
        for h in self.team_headers:
            self.compact_headers.remove(h)
            self.detailed_headers.remove(h)

        df = pd.read_csv(datadir + "Teams.csv")
        self.teams = dict(zip(df.Team_Id, df.Team_Name))
        self.seasons = pd.read_csv(datadir + "Seasons.csv")

        self.seeds = pd.read_csv(datadir + "TourneySeeds.csv")
        self.slots = pd.read_csv(datadir + "TourneySlots.csv")

    def getTeam(self, id):
        """Finds the team name of the given id.
        
        Retrieves the team name corresponding to the given team id from 
        the self.teams dataframe.

        Args:
            id: an integer representing the team id.

        Returns:
            The name of the team as a string.
        """
        return self.teams[id]

    def getYears(self, compact=True):
        """Gets a list of all the season years.
        
        Retrieves all the unique years from the self.seasons dataframe.
        
        Returns:
            A python list of years with game data.
        """
        years = self.seasons.Season.unique().tolist()
        return years if compact else years[years.index(2003):]

    def getSeeds(self, season):
        """Gets the seeds for each team in the given season.
        
        Retrives the team ids and their corresponding seeds for the given 
        season.

        Args:
            season: an integer for the year of the season to look up.

        Returns:
             A dict mapping team ids to their seeds for the season.
        """
        df = self.seeds.loc[self.seeds['Season'] == season]
        return dict(zip(df.Team, df.Seed))

    def getRegularGames(self, headers=None, season=None, compact=True):
        """Gets the dataframes for regular season data.
        
        Retrieves the game data from the regular seasons of the given years, 
        including the detailed game data if compact is False.
        See https://www.kaggle.com/c/march-machine-learning-mania-2017/data 
        for more details.

        Args:
            headers: a list of data headers to return
            season: an integer or list of integers for the years of regular 
            season game data to retrieve
            compact: if True only compact data columns will be returned

        Returns:
            A DataFrame containing the winning and losing teams,
            A DataFrame containing the relevant regular season data.
        """
        if headers is None:
            headers = self.compact_headers if compact else self.detailed_headers
        if season is None:
            return self.regular_results[self.team_headers], self.regular_results[headers]
        if type(season) is int: season = [season]
        results = self.regular_results.loc[self.regular_results['Season'].isin(list(season))]
        return results[self.team_headers], results[headers]

    def getTourneyGames(self, headers=None, season=None, compact=True):
        """Gets the dataframes for tourney game data.
        
        Retrieves the tourney data from the given years, including the 
        detailed tourney data if compact is False.
        See https://www.kaggle.com/c/march-machine-learning-mania-2017/data 
        for more details.

        Args:
            headers: a list of data headers to return
            season: an integer or list of integers for the years of regular 
            season game data to retrieve
            compact: if True only compact data columns will be returned

        Returns:
            A DataFrame containing the winning and losing teams,
            A DataFrame containing the relevant tourney data.
        """
        if headers is None:
            headers = self.compact_headers if compact else self.detailed_headers
        if season is None:
            return self.tourney_results[self.team_headers], self.tourney_results[headers]
        if type(season) is int: season = [season]
        results = self.tourney_results.loc[self.tourney_results['Season'].isin(list(season))]
        return results[self.team_headers], results[headers]

    def getFinalFour(self, season):
        """Gets the final four teams for a season

        Looks up the team ids that are competing on semifinals day (daynum=152) 
        of a given season from the tourney data.

        Args:
            season: an integer of the year of the season to look up

        Returns:
            A numpy array containing the four semifinalist team ids
        """
        semifinal_daynum = 152
        teams, days = self.getTourneyGames(headers=['Daynum'], season=season)
        return teams.loc[days['Daynum']==semifinal_daynum].to_numpy().flatten()
