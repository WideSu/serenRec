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

df = get_df('steam_reviews.json.gz')
df = get_df('steam_games.json.gz')
