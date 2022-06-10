import pandas as pd
import gzip

def parse(path):
  g = gzip.open(path, 'rb')
  for line in g:
    yield eval(line)

def get_df(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

steam_reviews = get_df('steam_reviews.json.gz')
steam_games = get_df('steam_games.json.gz')

steam_reviews.to_json('steam_reviews.json', orient='index')
steam_games.to_json('steam_games.json', orient='index')
