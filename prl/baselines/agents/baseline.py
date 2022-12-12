from datetime import datetime
from typing import List

import numpy as np
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as COLS
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env

from prl.baselines.agents.core.base_agent import Agent
from prl.baselines.agents.policies import StakeLevelImitationPolicy, CallingStation
from prl.baselines.supervised_learning.data_acquisition.core.parser import Blind, PlayerStack


class BaselineAgent(Agent):
    def reset(self, config):
        """Not needed, keep for consistency with interface"""
        pass

    def __init__(self, config, *args, **kwargs):
        """Wrapper around rllib policy of our baseline agent obtained from supervised learning of game logs"""
        super().__init__(config, *args, **kwargs)
        self._rllib_policy = config['rllib_policy']

    def act(self, observation):
        """Wraps rllib policy."""
        assert isinstance(observation, dict)
        return self._rllib_policy.compute_actions(observation)


def create_wrapped_environment(stacks):
    wrapped_env: AugmentObservationWrapper = init_wrapped_env(env_wrapper_cls=AugmentObservationWrapper,
                                                              stack_sizes=stacks,
                                                              multiply_by=1)
    return wrapped_env


def _get_player_stacks(obs, num_players, normalization_sum) -> List[PlayerStack]:
    """ Stacks at the beginning of every episode, not during or after."""
    player_stacks = []
    # Name the players, according to their offset to the button:
    # Button: Player_0
    # SB: Player_1
    # BB: Player_2
    # ...
    # Cut-off: Player_5
    pos_rel_to_observer = np.roll(['btn', 'sb', 'bb', 'utg', 'mp', 'co'], obs[COLS.Btn_idx])
    stacks = [COLS.Stack_p0,
              COLS.Stack_p1,
              COLS.Stack_p2,
              COLS.Stack_p3,
              COLS.Stack_p4,
              COLS.Stack_p5]
    for i in range(num_players):
        seat_display_name = pos_rel_to_observer[i]
        player_name = f'Player_{i}'
        stack = round(obs[stacks[i]] * normalization_sum, 2)
        player_stacks.append(PlayerStack(seat_display_name,
                                         player_name,
                                         stack))
    return player_stacks


def _get_blinds(obs, num_players, normalization_sum) -> List[Blind]:
    """
    observing player sits relative to button. this offset is given by
    # >>> obs[COLS.Btn_idx]
    the button position determines which players post the small and the big blind.
    For games with more than 2 pl. the small blind is posted by the player who acts after the button.
    When only two players remain, the button posts the small blind.
    """
    sb_name = "Player_1"
    bb_name = "Player_2"

    if num_players == 2:
        sb_name = "Player_0"
        bb_name = "Player_1"

    sb_amount = COLS.Small_blind * normalization_sum
    bb_amount = COLS.Big_blind * normalization_sum

    return [Blind(sb_name, 'small blind', sb_amount),
            Blind(bb_name, 'big_blind', bb_amount)]


if __name__ == '__main__':
    # configure run
    n_episodes = 100
    path_to_torch_model_state_dict = "/home/sascha/Documents/github.com/prl_baselines/prl/baselines/agents/experiments/ckpt.pt"
    env = create_wrapped_environment([1000, 1000])
    observation_space = env.observation_space
    action_space = env.action_space

    policy_config = {'path_to_torch_model_state_dict': path_to_torch_model_state_dict}
    baseline_policy = StakeLevelImitationPolicy(observation_space, action_space, policy_config)
    reference_policy = CallingStation(observation_space, action_space, policy_config)

    baseline_agent = BaselineAgent({'rllib_policy': baseline_policy})
    reference_agent = BaselineAgent({'rllib_policy': reference_policy})
    agents = [baseline_agent, reference_agent]
    n_agents = len(agents)

    variant = "HUNL"
    currency_symbol = "$"
    normalization_sum = float(
        sum([s.starting_stack_this_episode for s in env.env.seats])
    ) / env.env.N_SEATS

    num_players = 2
    for i in range(n_episodes):
        date = str(datetime.now())
        hand_id = i

        obs, _, done, _ = env.reset()
        # episode initialization
        blinds = _get_blinds(obs, num_players, normalization_sum)
        ante = obs[COLS.Ante] * normalization_sum
        player_stacks = _get_player_stacks(obs, num_players, normalization_sum)
        btn_idx = 3 if num_players > 2 else 0
        # game loop
        legal_moves = env.env.get_legal_actions()
        observation = {'obs': [obs], 'legal_moves': [legal_moves]}
        agent_idx = 0
        action_vec = agents[agent_idx].act(observation)
        # noinspection PyRedeclaration
        action = int(action_vec[0][0].numpy())
        while not done:
            obs, _, done, _ = env.step(action)
            legal_moves = env.env.get_legal_actions()
            observation = {'obs': [obs], 'legal_moves': [legal_moves]}
            agent_idx = (agent_idx + 1) % n_agents
            action_vec = agents[agent_idx].act(observation)
            action = int(action_vec[0][0].numpy())
            # todo update poker episode with action
            #  update actions_total
            # stage, player_name, action_type, raise_amount
            done = True
            # env.env.cards2str(env.env.get_hole_cards_of_player(0))
        # todo get board cards from env
        board = ''
        for card in env.env.board:
            board += env.env.cards2str(cards)
        #  get showdown hands
        #  get money collected
        # todo: make this the sanity check if very tight agent performs better vs calling station
        #  and the second ipynb should evaluate the agent purely in self play
