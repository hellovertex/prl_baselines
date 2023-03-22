# 1. load .ckpt file into baesline agent [x]
# 2. run games and collect rewards
import numpy as np
from prl.environment.Wrappers.base import ActionSpace

from prl.baselines.agents.tianshou_agents import ImitatorAgent
from prl.baselines.evaluation.v2.eval_agent import EvalAgentTianshou, EvalAgentRandom
from prl.baselines.examples.examples_tianshou_env import make_default_tianshou_env
import matplotlib.pyplot as plt

def load_imitation_agent(ckpt_dir):
    return ImitatorAgent(ckpt_dir,
                         flatten_input=False,
                         model_hidden_dims=(512,))


def run(max_episodes, agents, agent_names, pname):
    # 1. make env
    assert num_players == len(agents)
    path_out = "./baseline_stats"
    starting_stack = 5000
    env = make_default_tianshou_env(mc_model_ckpt_path=None,  # dont use mc
                                    stack_sizes=[starting_stack for _ in
                                                 range(len(agent_names))],
                                    agents=agent_names,
                                    num_players=len(agent_names))
    targets_rewards = []
    pidx = agent_names.index(pname)
    for _ in range(max_episodes):
        obs_dict = env.reset()
        obs = obs_dict['obs']
        agent_id = obs_dict['agent_id']
        legal_moves = obs_dict['mask']
        while True:
            i = agent_names.index(agent_id)
            action = agents[i].act([obs], legal_moves)
            if obs_dict['mask'][8] == 1:
                action = ActionSpace.NoOp
            obs_dict, rews, terminated, truncated, info = env.step(action)
            agent_id = obs_dict['agent_id']
            obs = obs_dict['obs']
            if terminated:
                targets_rewards.append(rews[pidx])
                break
    mean_rews = np.mean(targets_rewards)
    print(f'BB/H={mean_rews}, MBB/G={mean_rews * 1000}')
    return targets_rewards

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if __name__ == '__main__':
    max_episodes = 1000
    num_players = 6
    pname = 'AI_AGENT_v2'
    agent_names = [f'{pname}', '2', '3', '4', '5', '6']
    ckpt_dir = '/home/sascha/Documents/github.com/prl_baselines/data/05_train_results/from_gdrive/05_train_results-20230320T232447Z-001/05_train_results/NL50/player_pool/folds_from_top_players_with_randomized_hand/Top20Players_n_showdowns=5000/target_rounds=FTR/actions=ActionSpaceMinimal/512_1e-06/ckptdir/ckpt.pt'
    # if you want to load RL learners, use the loading_fns from train_eval.py
    agent = load_imitation_agent(ckpt_dir)
    agents = [EvalAgentTianshou(pname, agent)]
    agents += [EvalAgentRandom(f'p{i + 1}') for i in
               range(num_players - 1)]
    rews = run(max_episodes, agents, agent_names, pname)
    rews = np.cumsum(rews)
    fig = plt.figure(figsize=(8, 6), dpi=300)
    plt.xlabel('Hands played')
    plt.ylabel('Big Blinds won')
    ax = plt.gca()
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    plt.set_cmap('viridis')
    plt.grid()

    plt.axhline(0, color='black')

    plt.rcParams.update({'font.size': 12, 'font.family': 'Times New Roman'})
    plt.title('Winnings in Big Blinds per Game')
    plt.plot(np.arange(len(rews)), rews)
    plt.show()
