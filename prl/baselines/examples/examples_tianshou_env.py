from copy import deepcopy
from enum import IntEnum
from functools import partial
from typing import Tuple, List, Union, Optional, Dict

import gym
import numpy as np
from gym.spaces import Box
from pettingzoo import AECEnv
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import ObsType
from prl.environment.Wrappers.base import ActionSpace
from prl.environment.Wrappers.aoh import Positions6Max
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env
from prl.environment.Wrappers.vectorizer import AgentObservationType
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.env.venvs import SubprocVectorEnv
from prl.baselines.agents.mc_agent import MCAgent
from prl.baselines.agents.tianshou_policies import MultiAgentActionFlags
from prl.baselines.evaluation.core.experiment import ENV_WRAPPER
from omegaconf import DictConfig
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as cols
import logging

def get_cards(obs):
    one_hot_card_bit_range = slice(
        cols.First_player_card_0_rank_0, cols.First_player_card_1_suit_3 + 1
    )
    # observer cards are always at position 0
    return np.where(obs[one_hot_card_bit_range] == 1)


class RewardType(IntEnum):
    MBB = 0
    ABSOLUTE = 1


class TianshouEnvWrapper(AECEnv):
    """
    Multi Agent Environment that changes reset call such that
    observation, reward, done, info = reset()
        becomes
    observation = reset(),
    so that tianshou can parse observation properly.
    """

    def __init__(self,
                 env,
                 agents: List[str],
                 mc_ckpt_path: Optional[str],
                 reward_type: RewardType = RewardType.ABSOLUTE,
                 debug_reset_config_state_dict: Optional[Dict] = None):
        super().__init__()
        self.name = "env"
        self.reward_type = reward_type
        self.metadata = {'name': self.name}
        self.agents = agents
        self.possible_agents = self.agents[:]
        self.num_players = len(self.possible_agents)
        self.env_wrapped = env  # AugmentObservationWrapper
        self.debug_reset_config_state_dict = debug_reset_config_state_dict
        self.BIG_BLIND = self.env_wrapped.env.BIG_BLIND
        if mc_ckpt_path:
            self._mc_agent = MCAgent(
                ckpt_path=mc_ckpt_path, num_players=self.num_players)
        self._last_player_id = -1

        obs_space = Box(low=0.0, high=6.0, shape=(569,), dtype=np.float64)

        # if 'mask_legal_moves' in env_config:
        #     if env_config['mask_legal_moves']:
        observation_space = gym.spaces.Dict({
            'obs': obs_space,
            # do not change key-name 'obs' it is internally used by rllib (!)
            'action_mask': Box(low=0, high=1, shape=(3,), dtype=int)
            # one-hot encoded [FOLD, CHECK_CALL, RAISE]
        })
        self.observation_spaces = self._convert_to_dict(
            [observation_space for _ in range(self.num_agents)]
        )
        self.action_spaces = self._convert_to_dict(
            [self.env_wrapped.action_space for _ in range(self.num_agents)]
        )
        self.agent_map = {}
        for i in range(self.num_players):
            self.agent_map[i] = i
        self.btn = self.agent_map[0]
        self._last_obs = {}
        self.awaiting_noops = False

    def seed(self, seed: Optional[int] = None) -> None:
        self.seed = np.random.seed(seed)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent: str) -> Optional[ObsType]:
        return {"observation": self._last_obs[agent], "action_mask":
            self.next_legal_moves}
        # raise NotImplementedError

    def render(self) -> Union[None, np.ndarray, str, list]:
        return self.env_wrapped.render()  # returns None

    def state(self) -> np.ndarray:
        raise NotImplementedError

    def _convert_to_dict(self, list_of_list):
        return dict(zip(self.possible_agents, list_of_list))

    def _scale_rewards(self, rewards):
        if self.reward_type == RewardType.MBB:
            return [r / self.BIG_BLIND for r in rewards]
        if self.reward_type == RewardType.ABSOLUTE:
            return rewards
        else:
            raise NotImplementedError(f"Reward Type {self.reward_type} "
                                      f"not implemented.")

    def _int_to_name(self, ind):
        return self.possible_agents[ind]

    def reset(self,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None) -> None:
        if seed is not None:
            self.seed(seed=seed)
        reset_config = None
        self.awaiting_noops = False
        if options:
            if 'reset_config' in options:
                reset_config = options['reset_config']
        if self.debug_reset_config_state_dict is not None:
            reset_config = self.debug_reset_config_state_dict
        obs, rew, done, info = self.env_wrapped.reset(reset_config)
        player_id = self.agent_map[self.env_wrapped.env.current_player.seat_id]
        player = self._int_to_name(player_id)
        self.btn = self.agents[self.agent_map[0]]

        # self.starting_stacks = []
        # for i, seat in enumerate(self.env_wrapped.env.seats):
        #     pass
        self.agents = self.possible_agents[:]
        self.remaining = [a for a in self.agents]
        self.folded_players = []
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
        self.next_legal_moves = self.env_wrapped.get_legal_moves_extended()
        self.infos = self._convert_to_dict(
            [{"legal_moves": self.next_legal_moves,
              # "info": info} for _ in range(self.num_agents)]
              "info": []} for _ in range(self.num_agents)]
        )
        self._last_obs[self.agent_selection] = obs
        # self._last_obs = obs

    def action_was_fold(self, action):
        if isinstance(action, tuple):
            if action[0] == ActionSpace.FOLD:
                return True
        if action == ActionSpace.FOLD or np.array_equal(action, [ActionSpace.FOLD]):
            return True
        return False

    def action_was_noop(self, action):
        if isinstance(action, tuple):
            if action[0] == ActionSpace.NoOp:
                return True
        if action == ActionSpace.NoOp or np.array_equal(action, [ActionSpace.NoOp]):
            return True
        return False

    def step(self, action):
        # if isinstance(action, tuple):
        # if action == MultiAgentActionFlags.TriggerMC:
        #     action = self._mc_agent.compute_action(self._last_obs, self.next_legal_moves)
        if (
                self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        # only step the base env with real action
        # noops are used to distribute rewards and final observation to all players
        if action != ActionSpace.NoOp:
            try:
                assert not self.awaiting_noops
            except AssertionError:
                logging.warn(f"Tried to step environment with done="
                             f"{self.env_wrapped.done} and action"
                             f"{action} although awaiting_noops={self.awaiting_noops}."
                             f"Skipping Turn")
                return
            prev = self.env_wrapped.env.current_player.seat_id
            obs, rew, done, info = self.env_wrapped.step(action)
            # next_player_id = self.env_wrapped.env.current_player.seat_id
            next_player_id = (prev + 1) % self.num_players
            next_player_id = self.agent_map[next_player_id]
            next_player = self._int_to_name(next_player_id)
            # make observation for all players, so that everybody can see final
            #  cards
            # ~~roll button to correct position [BTN,...,] to [,...,BTN,...]~~
            # ~~roll relative to observer not to button~~
            # roll back to starting agent i.e. that reward of self.agents[0] is at 0
            rewards = np.roll(rew, -self.agent_map[Positions6Max.BTN])
            payouts = info['payouts']
            rpay = {}
            for k, v in payouts.items():
                rpay[self.agent_map[k]] = v
            info['payouts'] = payouts
            self.rewards = self._convert_to_dict(
                self._scale_rewards(rewards)
            )
            last_player_who_acted = self._int_to_name(self.agent_map[prev])
            # set final observations for each player
            if done:
                self._accumulate_rewards()
                # if env is not done, obs is for next player, otherwise obs is for same player
                for i in range(1, self.num_players):
                    prev_player_id = (prev - i) % self.num_players
                    prev_player_id = self.agent_map[prev_player_id]
                    prev_player_id = self._int_to_name(prev_player_id)
                    self._last_obs[prev_player_id] = self.env_wrapped.get_current_obs(
                        obs, backward_offset=i
                    )
                self._last_obs[last_player_who_acted] = obs
                nlm = np.zeros(len(ActionSpace))
                nlm[ActionSpace.NoOp] = 1
                self.next_legal_moves = nlm
                self.awaiting_noops = True
                # first, let the agent collect its observation that was last to act
                self.agent_selection = last_player_who_acted
            else:
                self._last_obs[next_player] = obs
                self.next_legal_moves = self.env_wrapped.get_legal_moves_extended()
                self.agent_selection = next_player

        if self.awaiting_noops:
            if len(self.remaining) > 0:
                # super calls self.observe(self.agent_selection) but its ok
                if self.action_was_noop(action):
                    self.remaining.pop(self.remaining.index(self.agent_selection))

            if len(self.remaining) <= 0:
                # All players have stepped their no-op: distribute rewards and show cards
                self.awaiting_noops = False
                # make sure last observation does not get rolled, so player can see cards
                # self.agent_selection = self._int_to_name(self.agent_map[prev])
                self.terminations = self._convert_to_dict(
                    [True for _ in range(self.num_agents)]
                )
                self.truncations = self._convert_to_dict(
                    [False for _ in range(self.num_agents)]
                )
                # move btn to next player
                shifted_indices = {}
                for rel_btn, agent_idx in self.agent_map.items():
                    shifted_indices[rel_btn] = (agent_idx + 1) % self.num_players
                self.agent_map = shifted_indices
            else:
                # collect no-ops to reveal cards to everybody
                try:
                    # first, let the agent collect its observation that was last to act
                    self.agent_selection = last_player_who_acted
                except Exception:
                    self.agent_selection = self.remaining[0]
                    if len(self.remaining) == 0:
                        self.awaiting_noops = False
        #else:
            # turns out tianshou does not utilize this:
            # if env is not done, obs is for next player, otherwise obs is for same player
            # for i in range(0, self.num_players - 1):
            #     prev_player_id = (prev - i) % self.num_players
            #     prev_player_id = self.agent_map[prev_player_id]
            #     prev_player_id = self._int_to_name(prev_player_id)
            #     self._last_obs[prev_player_id] = self.env_wrapped.get_current_obs(
            #         obs, backward_offset=i + 1
            # )

            # keep for reference but all players have to step no-op at the end. They will however not
            # if self.action_was_fold(action):
            #     self.folded_players.append(last_player_who_acted)
            # if next_player in self.folded_players:
            #     nlm = np.zeros(len(ActionSpace))
            #     nlm[ActionSpace.NoOp] = 1
            #     self.next_legal_moves = nlm
            # if self.agent_selection in

        self.infos = self._convert_to_dict(
            [{"legal_moves": self.next_legal_moves,
              # "info": info} for _ in range(self.num_agents)]
              "info": []} for _ in range(self.num_agents)]
        )
        # self._cumulative_rewards[self.agent_selection] = 0
        # self._accumulate_rewards()
        # self._deads_step_first()


class WrappedEnv(BaseWrapper):
    def seed(self, seed: Optional[int] = None) -> None:
        np.random.seed(seed)

    def __init__(self, env):
        # pettingzoo wrapper that copies agent selection, rewards etc from base env
        super().__init__(env)
        self.env = env


def make_default_tianshou_env(num_players=2,
                              agents=None,
                              stack_sizes=None,
                              blinds: List[int] = None,
                              mc_model_ckpt_path=None):
    starting_stack = 5000
    if blinds is None:
        blinds = [25, 50]
    if stack_sizes is None:
        stack_sizes = [starting_stack for _ in range(num_players)]
    if agents is None:
        agents = [f'p{i}' for i in range(num_players)]
    env_config = {"env_wrapper_cls": AugmentObservationWrapper,
                  # "stack_sizes": [100, 125, 150, 175, 200, 250],
                  "stack_sizes": stack_sizes,
                  "multiply_by": 1,
                  # use 100 for floats to remove decimals but we have int stacks
                  "scale_rewards": False,  # we do this ourselves
                  "blinds": blinds,
                  "agent_observation_mode": AgentObservationType.CARD_KNOWLEDGE}
    # env = init_wrapped_env(**env_config)
    # obs0 = env.reset(config=None)
    # AEC ENV
    env = TianshouEnvWrapper(env=make_env(env_config),
                             agents=agents,
                             mc_ckpt_path=mc_model_ckpt_path,
                             reward_type=RewardType.MBB)
    # to set seed as required by tianshou
    wrapped_env = WrappedEnv(env)
    wrapped_env = PettingZooEnv(wrapped_env)
    return wrapped_env


def make_env(cfg):
    return init_wrapped_env(**cfg)


def make_vectorized_pettingzoo_env(num_envs: int,
                                   single_env_config: Union[dict, DictConfig],
                                   agent_names: List[str],
                                   mc_model_ckpt_path: str,
                                   reward_type: RewardType = RewardType.MBB,
                                   debug_reset_config_state_dict: Optional[Dict] = None,
                                   ) -> Tuple[SubprocVectorEnv, PettingZooEnv]:
    assert len(agent_names) == len(single_env_config['stack_sizes'])
    env = TianshouEnvWrapper(env=make_env(single_env_config),
                             agents=agent_names,
                             reward_type=reward_type,
                             mc_ckpt_path=None,
                             debug_reset_config_state_dict=debug_reset_config_state_dict)

    def wrapped_env_fn():
        return PettingZooEnv(WrappedEnv(TianshouEnvWrapper(env=make_env(
            single_env_config),
            agents=agent_names,
            reward_type=reward_type,
            mc_ckpt_path=None,
            debug_reset_config_state_dict=debug_reset_config_state_dict)))

    wrapped_env = PettingZooEnv(WrappedEnv(env))
    venv = SubprocVectorEnv([wrapped_env_fn for _ in range(num_envs)])
    return venv, wrapped_env


# class TianshouWrappedSingleEnv(AugmentObservationWrapper):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     @override
#     def observation_space(self):
#         obs_space = Box(low=0.0, high=6.0, shape=(564,), dtype=np.float64)
#
#         # if 'mask_legal_moves' in env_config:
#         #     if env_config['mask_legal_moves']:
#         return gym.spaces.Dict({
#             'obs': obs_space,  # do not change key-name 'obs' it is internally used by rllib (!)
#             'action_mask': Box(low=0, high=1, shape=(3,), dtype=int)
#             # one-hot encoded [FOLD, CHECK_CALL, RAISE]
#         })


def make_vectorized_prl_env(num_envs: int,
                            single_env_config: dict,
                            agent_names: List[str],
                            ) -> Tuple[SubprocVectorEnv, ENV_WRAPPER]:
    assert len(agent_names) == len(single_env_config['stack_sizes'])
    single_env_config['disable_info'] = True  # drop info we do not need it with tianshou
    wrapped_env_fn = partial(make_env, single_env_config)
    wrapped_env = wrapped_env_fn()
    venv = SubprocVectorEnv([wrapped_env_fn for _ in range(num_envs)])
    return venv, wrapped_env
