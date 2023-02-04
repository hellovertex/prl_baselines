from prl.baselines.agents.tianshou_agents import BaselineAgent
from prl.baselines.evaluation.utils import get_reset_config, pretty_print
from prl.baselines.examples.examples_tianshou_env import make_default_tianshou_env

num_players = 3
verbose = True
hidden_dims = [256]
starting_stack = 20000
stack_sizes = [starting_stack for _ in range(num_players)]
agent_names = [f'p{i}' for i in range(num_players)]
# rainbow_config = get_rainbow_config(default_rainbow_params)
# RainbowPolicy(**rainbow_config).load_state_dict...
# env = get_default_env(num_players, starting_stack)
env = make_default_tianshou_env(mc_model_ckpt_path=None,  # dont use mc
                                agents=agent_names,
                                num_players=len(agent_names))
#  todo init from state dict and feed NN observations
board = '[Ks Kh Kd Kc 2s]'
player_hands = ['[Jh Jc]', '[4h 6s]', '[As 5s]']
state_dict = get_reset_config(player_hands, board)

model_ckpt_paths = [
    "/home/sascha/Documents/github.com/prl_baselines/data/new_snowie/with_folds/ckpt_dir/ilaviiitech_[256]_1e-06/ckpt.pt"]
if len(model_ckpt_paths) == 1:
    ckpt = model_ckpt_paths[0]
    model_ckpt_paths = [ckpt for _ in range(num_players)]
agents = [BaselineAgent(ckpt,
                        flatten_input=False,
                        num_players=num_players,
                        model_hidden_dims=hidden_dims) for ckpt in model_ckpt_paths]
assert len(agents) == num_players == len(stack_sizes)
options = {'reset_config': state_dict}
i = 0
for epoch in range(4):
    obs = env.reset(options=options)
    agent_id = obs['agent_id']
    legal_moves = obs['mask']
    obs = obs['obs']
    while True:
        i = agent_names.index(agent_id)
        action = agents[i].act(obs, legal_moves)
        print(f'AGNET_ID = {agent_id}')
        pretty_print(i, obs, action)
        obs_dict, cum_reward, terminated, truncated, info = env.step(action)
        rews = cum_reward
        agent_id = obs_dict['agent_id']
        print(f'AGENT_ID', agent_id)
        obs = obs_dict['obs']
        print(f'GOT REWARD {cum_reward}')
        if terminated:
            break



