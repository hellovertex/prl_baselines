from prl.baselines.agents.tianshou_agents import MCAgent
import pandas as pd

fpath = "/home/sascha/Documents/github.com/prl_baselines/data/03_preprocessed/0.25-0.50/6MAX_0.25-0.50_0.csv"
df = pd.read_csv(fpath,
                 sep=',',
                 dtype='float32',
                 encoding='cp1252')
legal_moves = [1, 1, 1, 1, 1, 1]  # 1 for each action
agent = MCAgent()
for i in range(len(df)):
    row = df.iloc[i].to_numpy()
    obs = row[1:]
    label = row[0]
    pred = agent.compute_action(obs, legal_moves)
