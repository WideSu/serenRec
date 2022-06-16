# import pandas as pd
# import gzip

# def parse(path):
#   g = gzip.open(path, 'rb')
#   for line in g:
#     yield eval(line)

# def get_df(path):
#   i = 0
#   df = {}
#   for d in parse(path):
#     df[i] = d
#     i += 1
#   return pd.DataFrame.from_dict(df, orient='index')

# steam_reviews = get_df('steam_reviews.json.gz')
# steam_games = get_df('steam_games.json.gz')

# steam_reviews.to_json('steam_reviews.json', orient='index')
# steam_games.to_json('steam_games.json', orient='index')

import gzip
import pandas as pd

class Loader():
    @staticmethod
    def __parse(path=""):
        """
            This static method is used to read content in gz files line by line
        :param path: The path of a gz file. (string, default value: "")
        :return: The string type data
        """
        g = gzip.open(path, 'rb')
        for line in g:
            yield eval(line)
    
    @staticmethod
    def get_df(path=""):
        """
            This static method is used to load from gz compress files and store data into pandas.DataFrame
        :param path: The path of a gz file. (string, default value: "")
        :return: The pandas.DataFrame class data
        """
        i = 0
        df = {}
        for d in Loader.__parse(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')
