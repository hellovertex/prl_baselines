from prl.baselines.evaluation.utils import get_default_env, print_player_stacks

n_players = 2
env = get_default_env(n_players)

# does the stack size decrease after reset?
obs, _, _, _ = env.reset()
print_player_stacks(obs, normalization_sum=env.normalization)
obs, reward, done, info = env.step(0)

print_player_stacks(obs, normalization_sum=env.normalization)
obs, _, _, _ = env.reset()
print_player_stacks(obs, normalization_sum=env.normalization)
# how to make it such that btn moves
# [BTN, SB, BB, UTG, MP, CO]
# [SB, BB, UTG, MP, CO, BTN]
map = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
num_players = 6
for i in range(10):
    shifted_map ={}
    for rel_btn, agent_idx in map.items():
        shifted_map[rel_btn] = (agent_idx + i) % num_players
    print(shifted_map)