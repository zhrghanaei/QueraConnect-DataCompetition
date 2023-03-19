import pandas as pd
df = pd.read_csv('Captain Tsubasa/train.csv')

### part 1 ###
print(df['playerId'].nunique())

### part 2 ###
print(df[df['outcome'] == 'گُل']['playerId'].mode())

### part 3 ###
from collections import defaultdict
d = defaultdict(int)
for player in df['playerId'].unique():
  df_p = df[df['playerId'] == player]
  d[player] = len(df_p[df_p['outcome'] == 'گُل']) / len(df_p)
print(max(d, key=d.get)+','+min(d, key=d.get))

### part 4 ###
import math
def dist(x, y):
  return int(math.sqrt(x**2 + y**2))

df['dist'] = df.apply(lambda row : dist(row['x'], row['y']), axis = 1)
max(df['dist'])