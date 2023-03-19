import pandas as pd
df = pd.read_csv('drive/MyDrive/football data/Captain Tsubasa/train.csv')

df["interferenceOnShooter"].fillna('متوسط', inplace = True)
predictions = model.predict(dict(df))
df['difficulty'] = pd.Series(np.squeeze(predictions))

players = list(df['playerId'].unique())
stats = pd.DataFrame(index = players, columns = ['score','goal_rate', 'out_rate', 'easy_goal', 'easy_out', 'hard_goal', 'hard_out'], dtype='int')
t = 0.5
for player in players:
  total = df[df.playerId == player]
  stats.loc[player]['goal_rate'] = len((total[total.outcome == 'گُل'])) / len(total)
  stats.loc[player]['out_rate'] = len((total[total.outcome != 'گُل'])) / len(total)
  stats.loc[player]['easy_goal'] = len((total[(df.difficulty < t) & (total.outcome == 'گُل')])) / len(total)
  stats.loc[player]['hard_goal'] = len((total[(df.difficulty >= t) & (total.outcome == 'گُل')])) / len(total)
  stats.loc[player]['easy_out'] = len((total[(df.difficulty < t) & (total.outcome != 'گُل')])) / len(total)
  stats.loc[player]['hard_out'] = len((total[(df.difficulty >= t) & (total.outcome != 'گُل')])) / len(total)

for player in players:
  stats.loc[player]['score'] = 50 * stats.loc[player]['goal_rate']    \
                              + 50 * stats.loc[player]['easy_goal']   \
                              + stats.loc[player]['hard_goal']  \
                              - 20 * stats.loc[player]['easy_out'] \
                              - stats.loc[player]['hard_out']

stats_sorted = stats.sort_values(by = ['score'], ascending=[False])
stats_sorted.head(30)