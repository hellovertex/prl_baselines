import os
import pprint
import time
from enum import IntEnum
from functools import partial
from random import random
from typing import Optional, Union, List, Tuple, Any, Dict

import gym
import numpy as np
import torch
from gym.spaces import Box
from pettingzoo import AECEnv
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import ObsType
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.base import ActionSpace
from prl.environment.Wrappers.utils import init_wrapped_env
from tianshou.data import Collector, Batch, PrioritizedVectorReplayBuffer
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.env.venvs import SubprocVectorEnv
from tianshou.policy import MultiAgentPolicyManager, RainbowPolicy, BasePolicy
from tianshou.trainer import offpolicy_trainer, OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter

from prl.baselines.cpp_hand_evaluator.monte_carlo import HandEvaluator_MonteCarlo
from prl.baselines.cpp_hand_evaluator.rank import dict_str_to_sk
from prl.baselines.agents.tianshou_agents import MCAgent
from prl.baselines.examples.rainbow_net import Rainbow
from prl.baselines.supervised_learning.models.nn_model import MLP


# todo implement this https://pettingzoo.farama.org/tutorials/tianshou/intermediate/
# todo: create one policy (actually zero because c51 already exists and the _mc_agent
#  does not have to be a policy after all
# todo: copy from policies.StakeLevelImitationPolicy.compute_action(obs:np.ndarray,...)
#  to create an agent that lives inside the environment


class MultiAgentActionFlags(IntEnum):
    """
    In the Baseline Agents, the Monte Carlo simulation takes a lot of time,
    so we want to parallelize its computations.
    Therefore, we include the MC simulator into the environment and let it be parallelized from
    tianshou by triggering the environment to compute the MC-based agent by itself, instead of
    getting it from the outside from an agent. This way when calling step with an MCTrigger - Action,
    the environment asks the BaselineAgent to compute its action as part of the env.step functionality.
    This parallelizes one MC-simulation per num_env. So we get maximum speedup by
    setting num_envs equal to the number of CPU cores available, e.g. 32.
    And can run 32 MC simulations at the same time -- the same number of env.step() we can call at
    the same time."""
    TriggerMC = 99


class MCAgentDummy:
    def act(self, _):
        return MultiAgentActionFlags.TriggerMC


class TianshouEnvWrapper(AECEnv):
    """
    Multi Agent Environment that changes reset call such that
    observation, reward, done, info = reset()
        becomes
    observation = reset(),
    so that tianshou can parse observation properly.
    """

    def __init__(self, env, agents: List[str]):
        super().__init__()
        self.name = "env"
        self.metadata = {'name': self.name}
        self.agents = agents
        self.possible_agents = self.agents[:]
        self.num_players = len(self.possible_agents)
        self.env_wrapped = env
        self.BIG_BLIND = self.env_wrapped.env.BIG_BLIND
        self._mc_agent = MCAgent(ckpt_path="/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/ckpt/ckpt.pt")
        self._last_player_id = -1

        obs_space = Box(low=0.0, high=6.0, shape=(564,), dtype=np.float64)

        # if 'mask_legal_moves' in env_config:
        #     if env_config['mask_legal_moves']:
        observation_space = gym.spaces.Dict({
            'obs': obs_space,  # do not change key-name 'obs' it is internally used by rllib (!)
            'action_mask': Box(low=0, high=1, shape=(3,), dtype=int)
            # one-hot encoded [FOLD, CHECK_CALL, RAISE]
        })
        self.observation_spaces = self._convert_to_dict(
            [observation_space for _ in range(self.num_agents)]
        )
        self.action_spaces = self._convert_to_dict(
            [self.env_wrapped.action_space for _ in range(self.num_agents)]
        )

    def seed(self, seed: Optional[int] = None) -> None:
        np.random.seed(seed)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent: str) -> Optional[ObsType]:
        return {"observation": self._last_obs, "action_mask": self.next_legal_moves}
        # raise NotImplementedError

    def render(self) -> Union[None, np.ndarray, str, list]:
        return self.env_wrapped.render()  # returns None

    def state(self) -> np.ndarray:
        raise NotImplementedError

    def _convert_to_dict(self, list_of_list):
        return dict(zip(self.possible_agents, list_of_list))

    def _scale_rewards(self, rewards):
        return [r / self.BIG_BLIND for r in rewards]

    def _int_to_name(self, ind):
        return self.possible_agents[ind]

    def roll_rewards(self, names: List[str], went_first: int, rewards: List[float], num_players: int):
        """
        Rolls indices from relative to button to relative to agent_list.
        In Backend the button is always at index 0, in PettingZoo env however, any player could be
        the button.
        @param names: Player Names as in PettingZoo wrapped PokerRL env.
        @param went_first: index of player in `agent_list` who started the round after env.reset()
        @param rewards: Payouts gotten from PokerRL backend. Button is always at index 0
        @param num_players: Total number of players at the table. Determines who goes first
        according to the following rule: If number of players is less than four,
        the button starts, otherwise the UTG in [Button, SmallBlind, BigBlind, UTG, MP, CO]
        starts.
        @return: The payouts dictionary with indices relative to the PettingZoo player name list,
        instead of relative to the button. The backend has the button at index 0/
        """
        # player who goes first, either button at index 0 or UTG at index 3
        offset = 0 if num_players < 4 else 3
        # Example 1: "Alice" goes first
        # - ["Bob", "Alice", "Tina"] --> ["Alice", "Tina", "Bob]
        # Example 2:  "Hans" goes first (implies Tina is button)
        # - ["Bob", "Alice", "Hans", "Tina"] --> ["Tina", "Bob", "Alice", "Hans"]
        # Example 3:  "Tina" goes first (implies Alice is button)
        # - ["Bob", "Alice", "Hans", "Stacy", "Tina"] --> ["Alice", "Hans", "Stacy", "Tina", "Bob"]
        return list(np.roll(rewards, offset - went_first))

    def reset(self,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None) -> None:
        if seed is not None:
            self.seed(seed=seed)
        obs, rew, done, info = self.env_wrapped.reset()
        player_id = (self._last_player_id + 1) % self.num_players
        player = self._int_to_name(player_id)
        self.goes_first = player_id
        self.agents = self.possible_agents[:]
        self.agent_selection = player
        self.rewards = self._convert_to_dict([0 for _ in range(self.num_agents)])
        self._cumulative_rewards = self._convert_to_dict(
            [0 for _ in range(self.num_agents)]
        )
        self.terminations = self._convert_to_dict(
            [False for _ in range(self.num_agents)]
        )
        self.truncations = self._convert_to_dict(
            [False for _ in range(self.num_agents)]
        )
        self.infos = self._convert_to_dict(
            [{"legal_moves": []} for _ in range(self.num_agents)]
        )
        legal_moves = np.array([0, 0, 0, 0, 0, 0])
        legal_moves[self.env_wrapped.env.get_legal_actions()] += 1
        if legal_moves[2] == 1:
            legal_moves[[3, 4, 5]] = 1
        self.next_legal_moves = legal_moves
        self._last_obs = obs

    def step(self, action):
        if action == MultiAgentActionFlags.TriggerMC:
            action = self._mc_agent.compute_action(self._last_obs, self.next_legal_moves)
        if (
                self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        obs, rew, done, info = self.env_wrapped.step(action)
        self._last_obs = obs
        next_player_id = (self._last_player_id + 1) % self.num_players
        next_player = self._int_to_name(next_player_id)
        if done:
            self.rewards = self._convert_to_dict(
                self._scale_rewards(self.roll_rewards(names=self.possible_agents,
                                                      went_first=self.goes_first,
                                                      rewards=rew,
                                                      num_players=self.num_players))
            )
            self.terminations = self._convert_to_dict(
                [True for _ in range(self.num_agents)]
            )
            self.truncations = self._convert_to_dict(
                [False for _ in range(self.num_agents)]
            )
        else:
            legal_moves = np.array([0, 0, 0, 0, 0, 0])
            legal_moves[self.env_wrapped.env.get_legal_actions()] += 1
            if legal_moves[2] == 1:
                legal_moves[[3, 4, 5]] = 1
            self.next_legal_moves = legal_moves
        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = next_player
        self._accumulate_rewards()
        self._deads_step_first()


class WrappedEnv(BaseWrapper):
    def seed(self, seed: Optional[int] = None) -> None:
        np.random.seed(seed)

    def __init__(self, env):
        super().__init__(env)
        self.env = env


env_config = {"env_wrapper_cls": AugmentObservationWrapper,
              # "stack_sizes": [100, 125, 150, 175, 200, 250],
              "stack_sizes": [10000, 10000],
              "multiply_by": 1,  # use 100 for floats to remove decimals but we have int stacks
              "scale_rewards": False,  # we do this ourselves
              "blinds": [50, 100]}
# env = init_wrapped_env(**env_config)
# obs0 = env.reset(config=None)
num_envs = 31


def make_env(cfg):
    return init_wrapped_env(**cfg)


agents = ["p0", "p1"]
env = TianshouEnvWrapper(make_env(env_config), agents)
wrapped_env_fn = partial(PettingZooEnv, WrappedEnv(env))
wrapped_env = PettingZooEnv(WrappedEnv(env))
venv = SubprocVectorEnv([wrapped_env_fn for _ in range(num_envs)])


def get_rainbow_config():
    # network
    classes = [ActionSpace.FOLD,
               ActionSpace.CHECK_CALL,  # CHECK IS INCLUDED
               ActionSpace.RAISE_MIN_OR_3BB,
               ActionSpace.RAISE_HALF_POT,
               ActionSpace.RAISE_POT,
               ActionSpace.ALL_IN]
    input_dim = 564
    hidden_dim = [512, 512]
    output_dim = len(classes)
    input_dim = 564  # hard coded for now -- very unlikely to be changed by me at any poiny in time
    device = "cuda"
    gamma = 0.99
    # device = "cpu"
    """
    Note: tianshou.policy.modelfree.c51.C51Policy.__init__ must move support to cuda if training on cuda
    self.support = torch.nn.Parameter(
            torch.linspace(self._v_min, self._v_max, self._num_atoms),
            requires_grad=False,
        ).cuda()
    """
    num_atoms = 51
    noisy_std = 0.1
    # Q_dict = V_dict = {'input_dim': input_dim,
    #                    "output_dim": output_dim,
    #                    "hidden_sizes": hidden_dim,
    #                    "device": device,
    #                    }
    # net = Net(state_shape=input_dim,
    #           action_shape=output_dim,
    #           hidden_sizes=hidden_dim,
    #           device=device,
    #           num_atoms=num_atoms,
    #           dueling_param=(Q_dict, V_dict)
    #           ).to(device)
    net = Rainbow(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_dim,
        device=device,
        num_atoms=num_atoms,
        noisy_std=noisy_std,
        is_dueling=True,
        is_noisy=True
    )
    optim = torch.optim.Adam(net.parameters(), lr=1e-6)
    # if running on GPU and we want to use cuda move model there
    return {'model': net,
            'optim': optim,
            'num_atoms': num_atoms,
            'v_min': -6,
            'v_max': 6,
            'estimation_step': 3,
            'target_update_freq': 500  # training steps
            }


class MCPolicy(BasePolicy):
    def __init__(self, observation_space=None, action_space=None):
        super().__init__(observation_space=observation_space,
                         action_space=action_space)

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, **kwargs: Any) -> Batch:
        nobs = len(batch.obs)
        return Batch(logits=None, act=[MultiAgentActionFlags.TriggerMC] * nobs, state=None)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        return {}


class TianshouCallingStation(BasePolicy):
    CHECK_CALL = 1

    def __init__(self, observation_space=None, action_space=None):
        super().__init__(observation_space=observation_space,
                         action_space=action_space)

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, **kwargs: Any) -> Batch:
        nobs = len(batch.obs)
        return Batch(logits=None, act=[self.CHECK_CALL] * nobs, state=None)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        return {}


rainbow_config = get_rainbow_config()
policy = MultiAgentPolicyManager([
    RainbowPolicy(**rainbow_config),
    RainbowPolicy(**rainbow_config),
#    MCPolicy()
], wrapped_env)  # policy is made from PettingZooEnv

buffer_size = 100000
obs_stack = 1
alpha = 0.5
beta = 0.4
beta_final = 1
beta_anneal_step = 5000000
weight_norm = True
# collector = Collector(policy, venv)
# t0 = time.time()
# result = collector.collect(n_episode=1000)
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
epoch = 10000
step_per_epoch = 10000
step_per_collect = 100
episode_per_test = 50
batch_size = 256
update_per_step = 0.1
learning_agent_ids = [0, 1]
eps_train = 0.2
eps_train_final = 0.05
eps_test = 0.0
no_priority = False


# def train_fn(epoch, env_step):
#     for aid in learning_agent_ids:
#         policy.policies[agents[aid]].set_eps(eps_train)

def train_fn(epoch, env_step, beta=beta):
    # nature DQN setting, linear decay in the first 1M steps
    if env_step <= 1e6:
        eps = eps_train - env_step / 1e6 * \
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


logdir = [".", "v3", "rainbow_vs_rainbow_heads_up"]


def save_best_fn(policy):
    model_save_path = os.path.join(
        *logdir, 'policy.pth'
    )
    for aid in learning_agent_ids:
        torch.save(
            policy.policies[agents[aid]].state_dict(), model_save_path
        )


win_rate = np.inf


def stop_fn(mean_rewards):
    return mean_rewards >= win_rate


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
                           save_checkpoint_fn=None,
                           resume_from_log=False,
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
