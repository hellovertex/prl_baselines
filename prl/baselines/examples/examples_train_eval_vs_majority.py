import multiprocessing
import os
import os
import pprint
import time
from pathlib import Path

import numpy as np
import torch
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from tianshou.data import Collector, PrioritizedVectorReplayBuffer
from tianshou.policy import RainbowPolicy, MultiAgentPolicyManager
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

from prl.baselines.agents.tianshou_agents import MajorityBaseline
from prl.baselines.agents.tianshou_policies import get_rainbow_config
from prl.baselines.examples.examples_tianshou_env import make_vectorized_prl_env, make_vectorized_pettingzoo_env


# todo move this script from prl.baselines to prl.reinforce

class Reward:
    def __init__(self):
        self.reward = 0


# train config
def train_eval(
        dir_suffix,
        num_players,
        device="cuda",
        buffer_size=100000,
        target_update_freq=500,
        obs_stack=1,
        alpha=0.5,
        beta=0.4,
        beta_final=1,
        beta_anneal_step=5000000,
        weight_norm=True,
        epoch=5_000,
        step_per_epoch=10_000,
        step_per_collect=100,
        episode_per_test=50,
        batch_size=256,
        update_per_step=0.1,
        eps_train=0.2,
        eps_train_final=0.05,
        eps_test=0.0,
        no_priority=False,
        load_ckpt=True):
    win_rate_early_stopping = np.inf,
    best_rew = -np.inf
    learning_agent_ids = [0]
    logdir = [".", "v7", "rainbow_vs_majority", dir_suffix]
    ckpt_save_path = os.path.join(
        *logdir, f'ckpt.pt'
    )
    # environment config
    starting_stack = 20000
    stack_sizes = [starting_stack for _ in range(num_players)]
    agents = [f'p{i}' for i in range(num_players)]
    sb = 50
    bb = 100
    env_config = {"env_wrapper_cls": AugmentObservationWrapper,
                  # "stack_sizes": [100, 125, 150, 175, 200, 250],
                  "stack_sizes": stack_sizes,
                  "multiply_by": 1,  # use 100 for floats to remove decimals but we have int stacks
                  "scale_rewards": False,  # we do this ourselves
                  "blinds": [sb, bb]}
    # env = init_wrapped_env(**env_config)
    # obs0 = env.reset(config=None)
    num_envs = 31
    input_folder = "/home/hellovertex/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_selected_players/with_folds/ckpt_dir"
    mc_model_ckpt_path = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/ckpt/ckpt.pt"
    player_dirs = [x[0] for x in
                   os.walk(input_folder)][1:]
    player_dirs = [pdir for pdir in player_dirs if not Path(pdir).stem == 'ckpt']
    ckpts = [pdir + '/ckpt.pt' for pdir in player_dirs]
    hidden_dims = [[256] if '[256]' in pname else [512] for pname in ckpts]
    majority_baseline = MajorityBaseline(ckpts,
                                         model_hidden_dims=hidden_dims,
                                         flatten_input=False,
                                         num_players=num_players)
    # for pdir in player_dirs:
    #     if not Path(pdir).stem == 'ckpt':
    #         # baseline analysis goes by position
    #         run_analysis_single_baseline(max_episodes=100,
    #                                      pname=Path(pdir).stem,
    #                                      ckpt_abs_fpath=pdir + '/ckpt.pt')
    #         # selected_player analysis goes by available .txt data
    ckpts = [pdir + '/ckpt.pt' for pdir in player_dirs]
    venv, wrapped_env = make_vectorized_pettingzoo_env(num_envs=num_envs,
                                                       single_env_config=env_config,
                                                       agent_names=agents,
                                                       mc_model_ckpt_path=mc_model_ckpt_path)
    params = {'device': device,
              'load_from_ckpt': ckpt_save_path,
              'lr': 1e-6,
              'num_atoms': 51,
              'noisy_std': 0.1,
              'v_min': -6,
              'v_max': 6,
              'estimation_step': 3,
              'target_update_freq': target_update_freq  # training steps
              }
    rainbow_config = get_rainbow_config(params)
    rainbow_policy = RainbowPolicy(**rainbow_config)
    if load_ckpt:
        try:
            rainbow_policy.load_state_dict(torch.load(ckpt_save_path, map_location=device))
        except FileNotFoundError:
            # initial training, no ckpt created yet, ignore silently
            pass
    # # 'load_from_ckpt_dir': None
    rainbow = RainbowPolicy(**rainbow_config)
    policy = MultiAgentPolicyManager([
        rainbow,
        majority_baseline,  # share weights
        #    MCPolicy()
    ], wrapped_env)  # policy is made from PettingZooEnv
    # policy = RainbowPolicy(**rainbow_config)

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

    max_reward = Reward()

    def reward_metric(rews):
        # The reward at index 0 is the reward relative to observer
        rew = np.mean(rews[:, learning_agent_ids[0]])
        if rew > max_reward.reward:
            max_reward.reward = rew
        return rew

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
            'net': policy.state_dict(),
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
    return max_reward.reward


def run_parallel(num_players):
    # target_update_frequencies = [5000, 50000, 100000, 1_000_000]
    # alphas = betas = [.4, .6, .8]
    # buffer_sizes = [10_000, 50_000, 100_000, 1_000_000]

    target_update_frequencies = [5000]
    alphas = betas = [.4]
    buffer_sizes = [50_000]
    curr_max_rew = 0
    max_freq = None
    max_alpha = None
    max_beta = None
    max_buffer_size = None

    for freq in target_update_frequencies:
        max_reward_sum = train_eval(f"num_players={num_players},targ_upd_freq={freq}",
                                    num_players=num_players,
                                    target_update_freq=freq)
        if max_reward_sum > curr_max_rew:
            curr_max_rew = max_reward_sum
            max_freq = freq

    curr_max_rew = 0
    max_reward_sum = 0

    for buffer_size in buffer_sizes:
        max_reward_sum = train_eval(f"num_players={num_players},buffer_size={buffer_size}",
                                    num_players=num_players,
                                    buffer_size=buffer_size,
                                    target_update_freq=max_freq)
        if max_reward_sum > curr_max_rew:
            curr_max_rew = max_reward_sum
            max_buffer_size = buffer_size

    curr_max_rew = 0
    max_reward_sum = 0

    for alpha in alphas:
        max_reward_sum = train_eval(f"num_players={num_players},alpha={alpha}",
                                    num_players=num_players,
                                    alpha=alpha,
                                    target_update_freq=max_freq,
                                    buffer_size=max_buffer_size)
        if max_reward_sum > curr_max_rew:
            curr_max_rew = max_reward_sum
            max_alpha = alpha

    curr_max_rew = 0
    max_reward_sum = 0

    for beta in betas:
        max_reward_sum = train_eval(f"num_players={num_players},beta={beta}",
                                    num_players=num_players,
                                    alpha=max_alpha,
                                    target_update_freq=max_freq,
                                    buffer_size=max_buffer_size,
                                    beta=beta)
        if max_reward_sum > curr_max_rew:
            curr_max_rew = max_reward_sum
            max_beta = beta

    print(f"Job Finished for num_players={num_players}, best parameters were:"
          f"target_update_freq={max_freq},\n "
          f"alpha={max_alpha},\n"
          f"beta={max_beta},\n"
          f"buffer_size={max_buffer_size}")


if __name__ == "__main__":
    # 1. todo fix reward rolling in AugmentObsWarpper
    num_players = [2, 6]
    start = time.time()
    p = multiprocessing.Pool()
    t0 = time.time()

    for x in p.imap_unordered(run_parallel, num_players):
        print(f'{x}. Took {time.time() - t0} seconds')
    print(f'Finished job after {time.time() - start} seconds.')

    p.close()
