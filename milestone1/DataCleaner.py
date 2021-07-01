import pandas as pd
import json

class DataCleaner:

    def __init__(self):
        self.movies = pd.read_csv("tmdb_5000_movies_train.csv")
        self.credits = pd.read_csv("tmdb_5000_credits_train.csv")
        self.merge()
        self.features = []

    def merge(self):
        self.movies = pd.merge(self.movies, self.credits, left_on='id', right_on='movie_id')
        self.movies = self.movies[:3799]

    def buildRelationalTables(self):

        data_keywords = list()
        data_genres = list()
        data_production_companies = list()
        data_production_countries = list()
        data_spoken_languages = list()
        data_cast = list()
        data_crew = list()
        for i in range(0, self.movies.shape[0]):

            keywords = json.loads(self.movies.iloc[i]['keywords'])
            production_companies = json.loads(self.movies.iloc[i]['production_companies'])
            production_countries = json.loads(self.movies.iloc[i]['production_countries'])
            genres = json.loads(self.movies.iloc[i]['genres'])
            spoken_languages = json.loads(self.movies.iloc[i]['spoken_languages'])
            cast = json.loads(self.movies.iloc[i]['cast'])
            crew = json.loads(self.movies.iloc[i]['crew'])
            vote_arverage = self.movies.iloc[i]['vote_average']

            for k in keywords:
                data_keywords.append([k['id'], vote_arverage])
            for p in production_companies:
                data_production_companies.append([p['id'], vote_arverage])
            for g in genres:
                data_genres.append([g['id'], vote_arverage])
            for c in production_countries:
                data_production_countries.append([c['iso_3166_1'], vote_arverage])
            for l in spoken_languages:
                data_spoken_languages.append([l['iso_639_1'], vote_arverage])
            for ca in cast:
                data_cast.append([ca['id'], vote_arverage])
            for cr in crew:
                data_crew.append([cr['id'], vote_arverage])
        return data_keywords, data_genres, data_production_companies, data_production_countries, data_spoken_languages, \
               data_cast, data_crew

    def dropUnnecessaryColumns(self):

        to_drop = ['homepage', 'id', 'original_title', 'overview', 'tagline','release_date']
        self.movies.drop(to_drop, inplace = True, axis = 1)

    def dropDuplicateColumns(self):

        to_drop = ['original_language', 'status', 'title_x', 'title_y', 'movie_id']
        self.movies.drop(to_drop, inplace=True, axis=1)

    def reformat(self):
        for i in self.movies.columns:
            self.features.append(i)
        #print(self.features)
        self.features[-2] = 'vote_count'
        self.features[-1] = 'vote_average'
        self.features[-3] = 'crew'
        self.features[-4] = 'cast'
        #print(self.features)
        self.movies=self.movies[self.features]
    def normalaize(self):
        for i in range(self.movies.shape[0]):
            for j in range(self.movies.shape[1]):
                if (max(self.movies.iloc[:, j]) - min(self.movies.iloc[:, j])) != 0:
                    self.movies.iloc[i, j] = (self.movies.iloc[i, j] - min(self.movies.iloc[:, j])) / (
                                max(self.movies.iloc[:, j]) - min(self.movies.iloc[:, j]))
                    #print(self.movies.iloc[i, j])

    def dropMissingRows(self):

        self.movies = self.movies[(self.movies['vote_count'] !=0).notnull() & (self.movies['runtime'] != 0).notnull()]

    def mapCategoricalFeature(self, i, featureLabel, comparableItem, featureRatio):
        feature = json.loads(self.movies.iloc[i][featureLabel])
        feature_rate = 0
        for g in feature:
            rate = float(featureRatio.loc[featureRatio['id'] == g[comparableItem], 'vote_arverage'])
            feature_rate += rate
        if len(feature) > 0:
            self.movies.at[i, featureLabel] = feature_rate / len(feature)
        else:
            self.movies.at[i, featureLabel] = 0

    def convertCategoricalFeaturestoNumerical(self):
        data_keywords, data_genres, data_production_companies, data_production_countries, data_spoken_languages, data_cast, data_crew = self.buildRelationalTables()
        keywords_ratio = pd.DataFrame(data_keywords, columns=['id', 'vote_arverage']).groupby('id', as_index=False).mean()
        genres_ratio = pd.DataFrame(data_genres, columns=['id', 'vote_arverage']).groupby('id', as_index=False).mean()
        production_companies_ratio = pd.DataFrame(data_production_companies, columns=['id', 'vote_arverage']).groupby('id', as_index=False).mean()
        production_countries_ratio = pd.DataFrame(data_production_countries, columns=['id', 'vote_arverage']).groupby('id', as_index=False).mean()
        spoken_languages_ratio = pd.DataFrame(data_spoken_languages, columns=['id', 'vote_arverage']).groupby('id', as_index=False).mean()
        cast_ratio = pd.DataFrame(data_cast, columns=['id', 'vote_arverage']).groupby('id', as_index=False).mean()
        crew_ratio = pd.DataFrame(data_crew, columns=['id', 'vote_arverage']).groupby('id', as_index=False).mean()
        for i in range(0, self.movies.shape[0]):
            self.mapCategoricalFeature(i, 'genres', 'id', genres_ratio)
            self.mapCategoricalFeature(i, 'keywords', 'id', keywords_ratio)
            self.mapCategoricalFeature(i, 'production_companies', 'id', production_companies_ratio)
            self.mapCategoricalFeature(i, 'production_countries', 'iso_3166_1', production_countries_ratio)
            self.mapCategoricalFeature(i, 'spoken_languages', 'iso_639_1', spoken_languages_ratio)
            self.mapCategoricalFeature(i, 'cast', 'id', cast_ratio)
            self.mapCategoricalFeature(i, 'crew', 'id', crew_ratio)

def runDataCleaner():
    d = DataCleaner()
    d.dropUnnecessaryColumns()
    d.dropDuplicateColumns()
    d.dropMissingRows()
    d.reformat()
    d.convertCategoricalFeaturestoNumerical()
    d.normalaize()
    d.movies.to_excel(r'numerical_data.xlsx')
