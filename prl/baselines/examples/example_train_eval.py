import os
import pprint
import time

import numpy as np
import torch
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from tianshou.data import Collector, PrioritizedVectorReplayBuffer
from tianshou.policy import MultiAgentPolicyManager, RainbowPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

from prl.baselines.agents.tianshou_policies import get_rainbow_config
from prl.baselines.examples.examples_tianshou_env import make_vector_env
# todo move this script from prl.baselines to prl.reinforce

# train config
device = "cuda"
buffer_size = 100000
obs_stack = 1
alpha = 0.5
beta = 0.4
beta_final = 1
beta_anneal_step = 5000000
weight_norm = True
epoch = 10000
step_per_epoch = 10000
step_per_collect = 100
episode_per_test = 50
batch_size = 256
update_per_step = 0.1
learning_agent_ids = [0]
eps_train = 0.2
eps_train_final = 0.05
eps_test = 0.0
no_priority = False
logdir = [".", "v3", "rainbow_vs_rainbow_heads_up"]
load_ckpt = True
ckpt_save_path = os.path.join(
        *logdir, f'ckpt.pt'
    )
win_rate_early_stopping = np.inf
best_rew = -np.inf
# environment config
num_players = 2
starting_stack = 20000
stack_sizes = [starting_stack for _ in range(num_players)]
agents = [f'p{i}' for i in range(num_players)]
env_config = {"env_wrapper_cls": AugmentObservationWrapper,
              # "stack_sizes": [100, 125, 150, 175, 200, 250],
              "stack_sizes": stack_sizes,
              "multiply_by": 1,  # use 100 for floats to remove decimals but we have int stacks
              "scale_rewards": False,  # we do this ourselves
              "blinds": [50, 100]}
# env = init_wrapped_env(**env_config)
# obs0 = env.reset(config=None)
num_envs = 31
mc_model_ckpt_path = os.environ["MC_MODEL_CKPT_PATH"]
venv, wrapped_env = make_vector_env(num_envs, env_config, agents, mc_model_ckpt_path)

params = {'device': device,
          'load_from_ckpt': ckpt_save_path,
          'lr': 1e-6,
          'num_atoms': 51,
          'noisy_std': 0.1,
          'v_min': -6,
          'v_max': 6,
          'estimation_step': 3,
          'target_update_freq': 500  # training steps
          }
rainbow_config = get_rainbow_config(params)
rainbow_policy = RainbowPolicy(**rainbow_config)
if load_ckpt:
    rainbow_policy.load_state_dict(torch.load(ckpt_save_path, map_location=device))
# 'load_from_ckpt_dir': None
policy = MultiAgentPolicyManager([
    RainbowPolicy(**rainbow_config),
    RainbowPolicy(**rainbow_config),
    #    MCPolicy()
], wrapped_env)  # policy is made from PettingZooEnv

buffer = PrioritizedVectorReplayBuffer(
    total_size=buffer_size,
    buffer_num=len(venv),
    ignore_obs_next=False,  # enable for framestacking
    save_only_last_obs=False,  # enable for framestacking
    stack_num=obs_stack,
    alpha=alpha,
    beta=beta,
    weight_norm=weight_norm
)
train_collector = Collector(policy, venv, buffer, exploration_noise=True)
test_collector = Collector(policy, venv, exploration_noise=True)


def train_fn(epoch, env_step, beta=beta):
    # linear decay in the first 10M steps
    if env_step <= 1e7:
        eps = eps_train - env_step / 1e7 * \
              (eps_train - eps_train_final)
    else:
        eps = eps_train_final
    for aid in learning_agent_ids:
        policy.policies[agents[aid]].set_eps(eps)
    if env_step % 1000 == 0:
        logger.write("train/env_step", env_step, {"train/eps": eps})
    if not no_priority:
        if env_step <= beta_anneal_step:
            beta = beta - env_step / beta_anneal_step * \
                   (beta - beta_final)
        else:
            beta = beta_final
        buffer.set_beta(beta)
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/beta": beta})


def test_fn(epoch, env_step):
    for aid in learning_agent_ids:
        policy.policies[agents[aid]].set_eps(eps_test)


def save_best_fn(policy):
    for aid in learning_agent_ids:
        model_save_path = os.path.join(
            *logdir, f'policy_{aid}.pth'
        )
        torch.save(
            policy.policies[agents[aid]].state_dict(), model_save_path
        )


def stop_fn(mean_rewards):
    return mean_rewards >= win_rate_early_stopping



def reward_metric(rews):
    # todo: consider computing the sum instead of single agent reward here
    return rews[:, learning_agent_ids[0]]


# watch agent's performance
# def watch():
#     print("Setup test envs ...")
#     policy.eval()
#     policy.set_eps(args.eps_test)
#     test_envs.seed(args.seed)
#     if args.save_buffer_name:
#         print(f"Generate buffer with size {args.buffer_size}")
#         buffer = PrioritizedVectorReplayBuffer(
#             args.buffer_size,
#             buffer_num=len(test_envs),
#             ignore_obs_next=True,
#             save_only_last_obs=True,
#             stack_num=args.frames_stack,
#             alpha=args.alpha,
#             beta=args.beta
#         )
#         collector = Collector(policy, test_envs, buffer, exploration_noise=True)
#         result = collector.collect(n_step=args.buffer_size)
#         print(f"Save buffer into {args.save_buffer_name}")
#         # Unfortunately, pickle will cause oom with 1M buffer size
#         buffer.save_hdf5(args.save_buffer_name)
#     else:
#         print("Testing agent ...")
#         test_collector.reset()
#         result = test_collector.collect(
#             n_episode=args.test_num, render=args.render
#         )
#     rew = result["rews"].mean()
#     print(f"Mean reward (over {result['n/ep']} episodes): {rew}")
#
# if args.watch:
#     watch()
#     exit(0)
# ======== tensorboard logging setup =========
log_path = os.path.join(*logdir)
writer = SummaryWriter(log_path)
# writer.add_text("args", str(args))
logger = TensorboardLogger(writer)


def save_checkpoint_fn(epoch: int,
                       env_step: int,
                       gradient_step: int) -> str:
    # for aid in learning_agent_ids:
    # assume learning agent is at index 0
    torch.save({
        'epoch': epoch,
        'net': policy.policies[agents[0]].state_dict(),
        'model': rainbow_config['model'].state_dict(),
        'env_step': env_step,
        'optim': rainbow_config['optim'].state_dict(),
    }, ckpt_save_path)
    return ckpt_save_path


# test train_collector and start filling replay buffer
train_collector.collect(n_step=batch_size * num_envs)
trainer = OffpolicyTrainer(policy=policy,
                           train_collector=train_collector,
                           test_collector=test_collector,
                           max_epoch=epoch,  # set stop_fn for early stopping
                           step_per_epoch=step_per_epoch,  # num transitions per epoch
                           step_per_collect=step_per_collect,  # step_per_collect -> update network -> repeat
                           episode_per_test=episode_per_test,  # games to play for one policy evaluation
                           batch_size=batch_size,
                           update_per_step=update_per_step,  # fraction of steps_per_collect
                           train_fn=train_fn,
                           test_fn=test_fn,
                           stop_fn=None,  # early stopping
                           save_best_fn=save_best_fn,
                           save_checkpoint_fn=save_checkpoint_fn,
                           resume_from_log=load_ckpt,
                           reward_metric=reward_metric,
                           logger=logger,
                           verbose=True,
                           show_progress=True,
                           test_in_train=False  # whether to test in training phase
                           )
result = trainer.run()
t0 = time.time()
pprint.pprint(result)
print(f'took {time.time() - t0} seconds')
# pprint.pprint(result)
# watch()

if __name__ == "__main__":
    pass