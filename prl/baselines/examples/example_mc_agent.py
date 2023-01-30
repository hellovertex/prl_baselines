import torch
from torch import softmax

from prl.baselines.agents.tianshou_agents import MCAgent
import pandas as pd

"""
"""

fpath = "/home/sascha/Documents/github.com/prl_baselines/data/03_preprocessed/0.25-0.50/6MAX_0.25-0.50_0.csv"
df = pd.read_csv(fpath,
                 sep=',',
                 dtype='float32',
                 encoding='cp1252')
legal_moves = [1, 1, 1, 1, 1, 1]  # 1 for each action
agent = MCAgent()
cummax = {0: 0,
          1: 0,
          2: 0,
          3: 0,
          4: 0,
          5: 0,
          6: 0}
n_obs = len(df)
for i in range(n_obs):
    print(f'Iterating row {i} of {n_obs}')
    row = df.iloc[i].to_numpy()
    obs = row[1:]
    label = row[0]
    pred = agent.compute_action(obs, legal_moves)
    cummax[pred] += torch.max(softmax(agent._logits, dim=1)).detach().numpy().item()
avg_acceptance = {}
for k, v in cummax.items():
    avg_acceptance[k] = v / n_obs
print(f'Average acceptance level over {n_obs} observations is per actions: '
      f'CHECK_CALL: {avg_acceptance[1]}\n'
      f'MIN_RAISE: {avg_acceptance[2]}\n'
      f'RAISE_HALF_POT: {avg_acceptance[3]}\n'
      f'RAISE_POT: {avg_acceptance[4]}\n'
      f'ALL_IN: {avg_acceptance[5]}\n')
