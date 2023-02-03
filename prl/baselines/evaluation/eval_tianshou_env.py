from prl.baselines.evaluation.utils import get_reset_config
from prl.baselines.examples.examples_tianshou_env import make_default_tianshou_env

mc_model_ckpt_path = "/home/sascha/Documents/github.com/prl_baselines/data/ckpt/ckpt.pt"
agents = ["Bob_0", "Tina_1", "Alice_2", "Hans_3"]
env = make_default_tianshou_env(mc_model_ckpt_path, num_players=len(agents))

board = '[6h Ts Td 9c Jc]'
player_hands = ['[6s 6d]', '[9s 9d]', '[Jd Js]', '[Ks Kd]']
state_dict = get_reset_config(player_hands, board)
options = {'reset_config': state_dict}
obs, rew, done, info = env.reset(options=options)
