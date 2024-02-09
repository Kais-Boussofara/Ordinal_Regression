import numpy as np
import pandas as pd
import copy

class get_season():
    def __init__(self, sport_type, year, print_info=False):
        self._sport_type = sport_type
        self._season = year
        self.frame = import_data(sport_type=sport_type, year=year)
        names = set(self.frame["home_team"].unique())
        names.update(self.frame["away_team"])
        self.names = sorted(list(names))
        self.M = len(names)
        self.T = len(self.frame)
        # self.names2ind = {n: i for i, n in enumerate(self.names)}
        self.names2ind = pd.Series(np.arange(self.M), index=self.names)
        
        result_cat, self.score_league = self.league_results("league") # get official scoring rule

        if print_info:
            print("-> imported {} season {}: number of teams= {}, number of games= {}".format(sport_type, year, self.M,
                                                                                          len(self.frame)))
            K_games = np.zeros(len(names), dtype=int)  # holder to the number of games
            for i, nn in enumerate(names):
                K_games[i] += ( (self.frame["home_team"] == nn).sum() + (self.frame["away_team"] == nn).sum())
            print("number of games per team = ", *set(K_games))

    def league_results(self, result_cat):
        # define the number of results
        # define the scoring rule when result_cat = "league"
        # "league" denotes the league-official results (e.g., merge OT and SO)
        # "full" denotes all possible distinguishable results
        if result_cat == "league":
            print("'standard' league's results will be used (depending on the season); e.g. 4 results for volleyball")
        if self._sport_type in {"SHL"}:
            if self._season < 1998:
                return 3, [0,1,2]    # scoring rule [2-1-0], no overtime
            elif self._season < 2010:
                return 5, [0,1,1,2,3]    # scoring rule [3-2-1-1-0]
            else:  # after 2010 shootouts are introduced
                if result_cat == "league":
                    return 4, [0,1,2,3]    # scoring rule [3-2-1-0]
                elif result_cat == "full":
                    return 6, [0,1,1,2,2,3]
        elif self._sport_type in {"NHL"}:
            if self._season < 2005:
                if result_cat == "league":
                    return 3, [0,1,2]
                elif result_cat == "full":
                    return 5, [0,0,1,2,2]
            else:
                if result_cat == "league":
                    return 4, [0,1,2,2]
                elif result_cat == "full":
                    return 6, [0,1,1,2,2,2]
        elif self._sport_type in {"SuperLega"}:
            if result_cat == "league":
                return 4, [0,1,2,3]
            elif result_cat == "full":
                return 6, [0,1,1,2,2,3]
        elif self._sport_type in {"EPL", "Championship", "LeagueOne", "LeagueTwo", "Bundesliga", "Bundesliga2", "LaLiga", "LaLiga2"}:
            return 3, [0, 1, 3]

        raise Exception("something wrong if I'm here")

    def define_results(self, result_cat="league"):
        ## we assign the numerical value (ordinal) to the result and we define the result_set which containt the description of the results
        if isinstance(result_cat, str):
            result_cat, xx = self.league_results(result_cat)

        self.frame["result"] = -1   # create new column

        if self._sport_type in {"EPL", "EPL_legacy",
                          "Championship", "LeagueOne", "LeagueTwo",
                          "Bundesliga", "Bundesliga_legacy",
                          "Bundesliga2", "LaLiga", "LaLiga2"}:
            result_set = ["loss", "draw", "win"]
            self.frame.loc[self.frame["home_score"] > self.frame["away_score"], "result"] = 2
            self.frame.loc[self.frame["home_score"] == self.frame["away_score"], "result"] = 1
            self.frame.loc[self.frame["home_score"] < self.frame["away_score"], "result"] = 0
        elif self._sport_type in {"NHL", "SHL"}:
            win = self.frame["home_score"] > self.frame["away_score"]
            loss = self.frame["home_score"] < self.frame["away_score"]
            draw = self.frame["home_score"] == self.frame["away_score"]
            OT = self.frame["SOOT"] == "OT"
            SO = self.frame["SOOT"] == "SO"
            if result_cat == 3:
                result_set = ["loss", "draw", "win"]
                self.frame.loc[win , "result"] = 2
                self.frame.loc[draw, "result"] = 1
                self.frame.loc[loss, "result"] = 0
            elif result_cat == 4:
                result_set = ["loss-RT", "loss-OTSO", "win-OTSO", "win-RT"]
                self.frame.loc[win & ~(OT | SO), "result"] = 3
                self.frame.loc[win & (OT | SO), "result"] = 2
                self.frame.loc[loss & (OT | SO), "result"] = 1
                self.frame.loc[loss & ~(OT | SO), "result"] = 0
            elif result_cat == 5:
                result_set = ["loss-RT", "loss-OT", "draw", "win-OT", "win-RT"]
                self.frame.loc[win & ~(OT | SO), "result"] = 4
                self.frame.loc[win & (OT | SO), "result"] = 3
                self.frame.loc[draw, "result"] = 2
                self.frame.loc[loss & (OT | SO), "result"] = 1
                self.frame.loc[loss & ~(OT | SO), "result"] = 0
            elif result_cat == 6:
                result_set = ["loss-RT", "loss-OT",  "loss-SO", "win-SO", "win-OT", "win-RT"]
                self.frame.loc[win & ~(OT | SO), "result"] = 5
                self.frame.loc[win & OT, "result"] = 4
                self.frame.loc[win & SO, "result"] = 3
                self.frame.loc[loss & SO, "result"] = 2
                self.frame.loc[loss & OT, "result"] = 1
                self.frame.loc[loss & ~(OT | SO), "result"] = 0
            else:
                raise Exception("undefined type of results for {}".format(self._sport_type))
        elif self._sport_type in {"SuperLega"}:
            if result_cat == 2:  # win-loss case
                result_set = ["loss", "win"]
                self.frame.loc[(self.frame["home_score"] == 3) & (self.frame["away_score"] == 0), "result"] = 1
                self.frame.loc[(self.frame["home_score"] == 3) & (self.frame["away_score"] == 1), "result"] = 1
                self.frame.loc[(self.frame["home_score"] == 3) & (self.frame["away_score"] == 2), "result"] = 1
                self.frame.loc[(self.frame["home_score"] == 2) & (self.frame["away_score"] == 3), "result"] = 0
                self.frame.loc[(self.frame["home_score"] == 1) & (self.frame["away_score"] == 3), "result"] = 0
                self.frame.loc[(self.frame["home_score"] == 0) & (self.frame["away_score"] == 3), "result"] = 0
            elif result_cat == 4:
                result_set = ["0-3 or 1-3", "2-3", "3-2", "3-1 or 3-0"]
                self.frame.loc[(self.frame["home_score"] == 3) & (self.frame["away_score"] == 0), "result"] = 3
                self.frame.loc[(self.frame["home_score"] == 3) & (self.frame["away_score"] == 1), "result"] = 3
                self.frame.loc[(self.frame["home_score"] == 3) & (self.frame["away_score"] == 2), "result"] = 2
                self.frame.loc[(self.frame["home_score"] == 2) & (self.frame["away_score"] == 3), "result"] = 1
                self.frame.loc[(self.frame["home_score"] == 1) & (self.frame["away_score"] == 3), "result"] = 0
                self.frame.loc[(self.frame["home_score"] == 0) & (self.frame["away_score"] == 3), "result"] = 0
            elif result_cat == 6:
                result_set = ["0-3", "1-3", "2-3", "3-2", "3-1", "3-0"]
                self.frame.loc[(self.frame["home_score"] == 3) & (self.frame["away_score"] == 0), "result"] = 5
                self.frame.loc[(self.frame["home_score"] == 3) & (self.frame["away_score"] == 1), "result"] = 4
                self.frame.loc[(self.frame["home_score"] == 3) & (self.frame["away_score"] == 2), "result"] = 3
                self.frame.loc[(self.frame["home_score"] == 2) & (self.frame["away_score"] == 3), "result"] = 2
                self.frame.loc[(self.frame["home_score"] == 1) & (self.frame["away_score"] == 3), "result"] = 1
                self.frame.loc[(self.frame["home_score"] == 0) & (self.frame["away_score"] == 3), "result"] = 0
            else:
                raise Exception("undefined type of results for {}".format(self._sport_type))
        else:
            raise Exception("variable 'results' undefined for ", sport_type)

        self.result_set = result_set

        ## give name to the numerical results
        self.frame["result_name"] = ""
        for kk, rr in enumerate(result_set):
            self.frame.loc[self.frame["result"] == kk, "result_name"] = rr
            if (self.frame["result"] == kk).sum() == 0:
                print("WARNING: result `{}` not found".format(rr))

    def basic_stats(self):
        if "result" not in self.frame.columns:
            self.define_results()

        K = len(self.result_set)    # number of possible outcomes
        W = [np.zeros((self.M, self.M)).astype(int) for rr in range(K)]   # list of matrices W
        for row in self.frame.iterrows():
            rec = row[1]
            m = self.names2ind[rec["home_team"]]
            n = self.names2ind[rec["away_team"]]
            rr = rec["result"]
            W[rr][m, n] += 1

        # here we use symmetry of the result (when order of the home-away team does not matter)
        WW = copy.deepcopy(W)
        for rr in range(K):
            W[rr] += WW[K-1-rr].T
        self.W = W

        goal_for = np.zeros(self.M).astype(int)
        goal_against = np.zeros(self.M).astype(int)
        for i, team in enumerate(self.names):
            goal_for[i] = self.frame[self.frame["home_team"] == team]["home_score"].sum()
            goal_for[i] += self.frame[self.frame["away_team"] == team]["away_score"].sum()
            goal_against[i] = self.frame[self.frame["home_team"] == team]["away_score"].sum()
            goal_against[i] += self.frame[self.frame["away_team"] == team]["home_score"].sum()

        # create the summary for each team: number of results of each type + goal for/againt + goals difference
        df = pd.DataFrame(index=self.names,
                          columns=self.result_set + ["goal_for", "goal_against", "goal_difference"])
        for rr, rr_name in enumerate(self.result_set):
            df[rr_name] = W[rr].sum(axis=1)

        df["goal_for"] = goal_for
        df["goal_against"] = goal_against
        df["goal_difference"] = goal_for - goal_against
        self.stats = df

    def regression_format(self):
        ## organize data in the format ready for regression pair (X,y)
        ## where X has "1" for the home teams and "-1" for away teams, and y has the results
        if "result" not in self.frame.columns:
            self.define_results()
    
        ii = self.names2ind[self.frame["home_team"].values].values
        jj = self.names2ind[self.frame["away_team"].values].values
        
        XX = np.zeros((self.T, self.M))
        XX[np.arange(self.T), ii] = 1
        XX[np.arange(self.T), jj] = -1
        yy = self.frame["result"].values
        
        self.X = XX
        self.y = yy 


    ######################################################################
        
    def rank_teams(self, xi=None, rank_by="best"):
    # rank the teams using the weights xi associated with the results inself.stats
        if not hasattr(self, "stats"):
            self.basic_stats()
        if xi is None:
            xi = self.score_league
        if len(xi) != len(self.W):
            raise Exception("scoring rule xi must have the same length as the list of matrices W")
        if any(np.diff(xi) < 0):
            print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
            print("Warning: we expect the scoring values to be non-decresing")
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        rank_frame = self.stats.copy()

        # calculate the score (for the best team)

        rank_frame.insert(loc=0, column="point", value=0)
        for k, rr in enumerate(self.result_set):
            rank_frame["point"] += xi[k] * rank_frame[rr]

        # calculate the score (for the worst team)
        rank_frame.insert(loc=1, column="point_worst", value=0)
        K = len(self.result_set)
        for k, rr in enumerate(self.result_set):
            rank_frame["point_worst"] += xi[K-1-k] * rank_frame[rr]

        #  ranking
        if rank_by == "best":
            rank_frame.sort_values(by=["point", "goal_difference", "goal_for"], inplace=True, ascending=False)
        elif rank_by == "worst":
            rank_frame.sort_values(by=["point_worst", "goal_difference", "goal_for"], inplace=True, ascending=True)

        return rank_frame


########################################################################
def import_data(sport_type, year):
    # sport_type : "NHL', "EPL"
    if isinstance(year, list):
        imported_data = [import_data(sport_type, yy) for yy in year]
        return imported_data

    if sport_type == "EPL" and year < 1993:
        sport_type = "EPL_legacy"
    elif sport_type == "Bundesliga" and year < 1993:
        sport_type = "Bundesliga_legacy"

    dir = "~/PycharmProjects/DATA/" + sport_type + "/"
    if sport_type == "NHL":
        columns_to_drop = ["Att.", "LOG", "Notes"]
        name_changer = {"Visitor": "away_team", "Home": "home_team", "Date":"date"}
        name_changer["G"] = "away_score"
        name_changer["G.1"] = "home_score"
        name_changer["Unnamed: 5"] = "SOOT"

        day_first_indicator = False
        year_first_indicator = True
    elif sport_type in {"EPL", "EPL_legacy", 
                        "Championship", "LeagueOne", "LeagueTwo", 
                        "Bundesliga", "Bundesliga_legacy", 
                        "Bundesliga2", "LaLiga", "LaLiga2"}:
        if sport_type in {"EPL_legacy"}:
            columns_to_drop = ["round"]
        else:
            columns_to_drop = []  # ["Div", "Referee", "FTR", "HTHG", "HTAG", "HTR"]
        name_changer = {"AwayTeam": "away_team", "HomeTeam": "home_team", "Date": "date"}
        name_changer["FTAG"] = "away_score"
        name_changer["FTHG"] = "home_score"

        day_first_indicator = True
        year_first_indicator = False
    elif sport_type == "NFL":
        columns_to_drop = ["Unnamed: 7", "YdsW", "TOW", "YdsL", "TOL", "Time"]
        name_changer = {"Winner/tie": "home_team", "Loser/tie": "away_team"}
        name_changer["Pts"] = "home_score"
        name_changer["Pts.1"] = "away_score"
        name_changer["Unnamed: 5"] = "HA-switch"  # the sign "@" indicates that the winner is away team

        day_first_indicator = False
        year_first_indicator = True
    elif sport_type == "SuperLega":
        columns_to_drop = []
        name_changer = {}
        day_first_indicator = False
        year_first_indicator = True
    elif sport_type == "SHL":
        columns_to_drop = []
        name_changer = {}
        day_first_indicator = False
        year_first_indicator = True
    else:
        raise Exception("sport:" + sport_type + " is undefined")

    save_file_name_template = sport_type + "-{:4d}-{:4d}.csv".format(year, year + 1)
    df = pd.read_csv(dir + save_file_name_template)
    df.drop(columns=columns_to_drop, inplace=True)
    df.rename(columns=name_changer, inplace=True)
    df.fillna("", inplace=True)

    unique_players_home = df["home_team"].unique()
    unique_players_away = df["away_team"].unique()
    if len(unique_players_away) != len(unique_players_home):
        raise Exception("numbers of unique home- and away- teams are not the same")
    M = len(unique_players_away)

    ## sport-specific modifications of the frame
    if sport_type in {"NHL", "SHL"}:
        goals = df["home_score"]
        ind_drop = list(goals[goals == ""].index)  # find unspecified goal (game not played)
        df.drop(index=ind_drop, inplace=True)
    elif sport_type == "NFL":
        ind_drop = list(df.loc[df["Week"] == "Week"].index)  # find empty lines (weeks separators)
        df.drop(index=ind_drop, inplace=True)
        ind_drop2 = list(
            range(df.loc[df["Date"] == "Playoffs"].index[0], df.index[-1] + 1))  # remove from playoffs to the end
        df.drop(index=ind_drop2, inplace=True)

        switch_ind = df.loc[df["HA-switch"] == "@"].index
        player_tmp = df["home_team"][switch_ind]
        df["home_team"][switch_ind] = df["away_team"][switch_ind]
        df["away_team"][switch_ind] = player_tmp

        goals_tmp = df["home_goals"][switch_ind]
        df["home_goals"][switch_ind] = df["away_goals"][switch_ind]
        df["away_goals"][switch_ind] = goals_tmp

        df.drop(columns=["HA-switch"], inplace=True)
    elif sport_type == "Bundesliga_legacy":
        name_changer = {"Meidericher SV": "MSV Duisburg",
                        "Eintracht Frankfurt": "Frankfurter SG Eintracht",
                        "Bayer Leverkusen": "SV Bayer 04 Leverkusen"}
        df.replace(name_changer, inplace=True)

    if (2*len(df)) % (M) > 0:
        raise Exception("problem: number of games x2 should be a multiple of number of teams")
    ## calculate and insert the time_stamp
    z = pd.to_datetime(df["date"], dayfirst=day_first_indicator, yearfirst=year_first_indicator)
    day_stamp = (z - z[0]).dt.days
    if any(day_stamp < 0):
        raise Exception("we have a problem: the date difference cannot be negative")
    if any(np.diff(day_stamp) < 0):
        print("Warning: {}, {}; the dates are not increasing linearly (games cancelled and played later?)".format(sport_type,year))
    df.insert(5, "time_stamp", day_stamp)

    return df
