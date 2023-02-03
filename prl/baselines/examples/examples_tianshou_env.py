from enum import IntEnum
from functools import partial
from random import random
from typing import Tuple, List, Union, Optional

import gym
import numpy as np
import torch
from gym.spaces import Box
from pettingzoo import AECEnv
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import ObsType
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as cols
from prl.environment.Wrappers.base import ActionSpace
from prl.environment.Wrappers.utils import init_wrapped_env
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.env.venvs import SubprocVectorEnv
from torch import softmax

from prl.baselines.agents.tianshou_policies import MultiAgentActionFlags
from prl.baselines.cpp_hand_evaluator.monte_carlo import HandEvaluator_MonteCarlo
from prl.baselines.cpp_hand_evaluator.rank import dict_str_to_sk
from prl.baselines.supervised_learning.models.nn_model import MLP


class RewardType(IntEnum):
    MBB = 0
    ABSOLUTE = 1


IDX_C0_0 = 167  # feature_names.index('0th_player_card_0_rank_0')
IDX_C0_1 = 184  # feature_names.index('0th_player_card_1_rank_0')
IDX_C1_0 = 184  # feature_names.index('0th_player_card_1_rank_0')
IDX_C1_1 = 201  # feature_names.index('1th_player_card_0_rank_0')
IDX_BOARD_START = 82  #
IDX_BOARD_END = 167  #
CARD_BITS_TO_STR = np.array(['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c'])
BOARD_BITS_TO_STR = np.array(['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
                              'h', 'd', 's', 'c', '2', '3', '4', '5', '6', '7', '8', '9', 'T',
                              'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c', '2', '3', '4', '5', '6',
                              '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c', '2',
                              '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h',
                              'd', 's', 'c', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J',
                              'Q', 'K', 'A', 'h', 'd', 's', 'c'])
RANK = 0
SUITE = 1


class MCAgent:

    def __init__(self, ckpt_path, num_players):
        """ckpt path may be something like ./ckpt/ckpt.pt"""
        self._mc_iters = 10000
        self.num_players = num_players
        self.tightness = .1  # percentage of hands played, "played" meaning not folded immediately
        self.acceptance_threshold = 0.5  # minimum certainty of probability of network to perform action
        self._card_evaluator = HandEvaluator_MonteCarlo()
        self.load_model(ckpt_path)

    def load_model(self, ckpt_path):
        input_dim = 564
        classes = [ActionSpace.FOLD,
                   ActionSpace.CHECK_CALL,  # CHECK IS INCLUDED in CHECK_CALL
                   ActionSpace.RAISE_MIN_OR_3BB,
                   ActionSpace.RAISE_HALF_POT,
                   ActionSpace.RAISE_POT,
                   ActionSpace.ALL_IN]
        hidden_dim = [512, 512]
        output_dim = len(classes)
        net = MLP(input_dim, output_dim, hidden_dim)
        # if running on GPU and we want to use cuda move model there
        # use_cuda = torch.cuda.is_available()
        # if use_cuda:
        #     net = net.cuda()
        self._model = net
        ckpt = torch.load(ckpt_path,
                          map_location=torch.device('cpu'))
        self._model.load_state_dict(ckpt['net'])
        self._model.eval()
        return net

    def card_bit_mask_to_int(self, c0: np.array, c1: np.array, board_mask: np.array) -> Tuple[List[int], List[int]]:
        c0_1d = dict_str_to_sk[CARD_BITS_TO_STR[c0][RANK] + CARD_BITS_TO_STR[c0][SUITE]]
        c1_1d = dict_str_to_sk[CARD_BITS_TO_STR[c1][RANK] + CARD_BITS_TO_STR[c1][SUITE]]
        board = BOARD_BITS_TO_STR[board_mask.astype(bool)]
        # board = array(['A', 'c', '2', 'h', '8', 'd'], dtype='<U1')
        board_cards = []
        for i in range(0, sum(board_mask) - 1, 2):  # sum is 6,8,10 for flop turn river resp.
            board_cards.append(dict_str_to_sk[board[i] + board[i + 1]])

        return [c0_1d, c1_1d], board_cards

    def look_at_cards(self, obs: np.array) -> Tuple[List[int], List[int]]:
        c0_bits = obs[IDX_C0_0:IDX_C0_1].astype(bool)
        c1_bits = obs[IDX_C1_0:IDX_C1_1].astype(bool)
        board_bits = obs[IDX_BOARD_START:IDX_BOARD_END].astype(int)  # bit representation
        return self.card_bit_mask_to_int(c0_bits, c1_bits, board_bits)

    def _compute_action(self,
                        obs: Union[List, np.ndarray]):
        hero_cards_1d, board_cards_1d = self.look_at_cards(obs)
        # from cards get winning probabilityx
        mc_dict = self._card_evaluator.run_mc(hero_cards_1d,
                                              board_cards_1d,
                                              self.num_players,
                                              n_iter=self._mc_iters)
        # {won: 0, lost: 0, tied: 0}[
        win_prob = float(mc_dict['won'] / self._mc_iters)
        # todo: replace win_prob < .5
        total_to_call = obs[cols.Total_to_call]
        # if we have negative EV on calling/raising, we fold with high probability
        potsize = sum([
            obs[cols.Curr_bet_p1],
            obs[cols.Curr_bet_p2],
            obs[cols.Curr_bet_p3],
            obs[cols.Curr_bet_p4],
            obs[cols.Curr_bet_p5],
        ]) + obs[cols.Pot_amt]
        pot_odds = total_to_call / (potsize + total_to_call)
        # ignore pot odds preflop, we marginalize fold probability via acceptance threshold
        if not obs[cols.Round_preflop]:
            # fold when bad pot odds are not good enough according to MC simulation
            if win_prob < pot_odds:
                if random() > self.tightness:  # tightness is equal to % of hands played, e.g. 0.15
                    return ActionSpace.FOLD.value
        certainty = torch.max(softmax(self._logits, dim=1)).detach().numpy().item()
        # if the probability is high enough, we take the action suggested by the neural network
        if certainty > self.acceptance_threshold:
            assert len(self._predictions == 1)
            prediction = self._predictions[0]
            # if we are allowed, we raise otherwise we check
            # todo: should we add pseudo harmonic mapping here with train/eval modes?
            if self.next_legal_moves[min(int(prediction), 2)]:
                return int(prediction)
            elif ActionSpace.CHECK_CALL in self.next_legal_moves:
                return ActionSpace.CHECK_CALL.value
            else:
                return ActionSpace.FOLD.value
        else:
            return ActionSpace.FOLD.value
        # 3. pick argmax only if p(argmax) > 1-acceptance
        # 4. start with acceptance 1 and decrease until
        # 5. tighntess(mc_agent) == tightness(players) = n_hands_played/n_total_hands
        # 6. n_hands_played should be hands that are not immediately folded preflop
        # 7. gotta compute them from the players datasets

    def compute_action(self, obs: np.ndarray, legal_moves) -> int:
        self.next_legal_moves = legal_moves
        self._logits = self._model(torch.Tensor(np.array([obs])))
        self._predictions = torch.argmax(self._logits, dim=1)
        action = self._compute_action(obs)
        return action

    def act(self, obs, legal_moves):
        """Wrapper for compute action only when evaluating in poke--> single env"""
        if type(obs) == dict:
            legal_moves = np.array([0, 0, 0, 0, 0, 0])
            legal_moves[obs['legal_moves'][0]] += 1
            if legal_moves[2] == 1:
                legal_moves[[3, 4, 5]] = 1
            obs = obs['obs']
            if type(obs) == list:
                obs = np.array(obs)[0]
        return self.compute_action(obs, legal_moves)


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
                 reward_type: RewardType,
                 mc_ckpt_path: str):
        super().__init__()
        self.name = "env"
        self.reward_type = reward_type
        self.metadata = {'name': self.name}
        self.agents = agents
        self.possible_agents = self.agents[:]
        self.num_players = len(self.possible_agents)
        self.env_wrapped = env
        self.BIG_BLIND = self.env_wrapped.env.BIG_BLIND
        self._mc_agent = MCAgent(
            ckpt_path=mc_ckpt_path, num_players=self.num_players)
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
        if self.reward_type == RewardType.MBB:
            return [r / self.BIG_BLIND for r in rewards]
        if self.reward_type == RewardType.ABSOLUTE:
            return rewards
        else:
            raise NotImplementedError(f"Reward Type {self.reward_type} "
                                      f"not implemented.")

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


def make_env(cfg):
    return init_wrapped_env(**cfg)


def make_vector_env(num_envs: int,
                    single_env_config: dict,
                    agent_names: List[str],
                    mc_model_ckpt_path: str,
                    reward_type: RewardType = RewardType.MBB,
                    ) -> Tuple[SubprocVectorEnv, PettingZooEnv]:
    assert len(agent_names) == len(single_env_config['stack_sizes'])
    env = TianshouEnvWrapper(env=make_env(single_env_config),
                             agents=agent_names,
                             reward_type=reward_type,
                             mc_ckpt_path=mc_model_ckpt_path)
    wrapped_env_fn = partial(PettingZooEnv, WrappedEnv(env))
    wrapped_env = PettingZooEnv(WrappedEnv(env))
    venv = SubprocVectorEnv([wrapped_env_fn for _ in range(num_envs)])
    return venv, wrapped_env
