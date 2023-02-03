from prl.baselines.examples.examples_tianshou_env import make_default_tianshou_env

mc_model_ckpt_path = "/home/sascha/Documents/github.com/prl_baselines/data/ckpt/ckpt.pt"
env = make_default_tianshou_env(mc_model_ckpt_path, num_players=2)