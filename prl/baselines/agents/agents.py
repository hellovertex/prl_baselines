import os
from datetime import datetime
from typing import List, Dict

import gin
import numpy as np
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as COLS
from prl.environment.Wrappers.utils import init_wrapped_env
from prl.environment.steinberger.PokerRL import Poker

from prl.baselines.agents.core.base_agent import Agent
from prl.baselines.agents.policies import StakeLevelImitationPolicy, CallingStation
from prl.baselines.supervised_learning.data_acquisition.core.parser import Blind, PlayerStack, ActionType, Action, \
    PokerEpisode, PlayerWithCards, PlayerWinningsCollected


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


POSITIONS = ['btn', 'sb', 'bb', 'utg', 'mp', 'co']
STAGES = [Poker.PREFLOP, Poker.FLOP, Poker.TURN, Poker.RIVER]
ACTION_TYPES = [ActionType.FOLD, ActionType.CHECK_CALL, ActionType.RAISE]


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
        player_name = pos_rel_to_observer[i]
        seat_display_name = player_name
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


class Dummy:
    def f(self):
        print('hello dummy')


def make_agents(env, path_to_torch_model_state_dict):
    policy_config = {'path_to_torch_model_state_dict': path_to_torch_model_state_dict}
    baseline_policy = StakeLevelImitationPolicy(env.observation_space, env.action_space, policy_config)
    reference_policy = CallingStation(env.observation_space, env.action_space, policy_config)

    baseline_agent = BaselineAgent({'rllib_policy': baseline_policy})
    reference_agent = BaselineAgent({'rllib_policy': reference_policy})
    return [baseline_agent, baseline_agent]


def _get_winners(showdown_players: List[PlayerWithCards], payouts: dict) -> List[PlayerWithCards]:
    winners = []
    for pid, money_won in payouts.items():
        for p in showdown_players:
            # look for winning player in showdown players and append its stats to winners
            if POSITIONS[pid] == p.name:
                winners.append(PlayerWithCards(name=p.name,
                                               cards=p.cards))
    return winners


def _get_money_collected(payouts: Dict[int, str]) -> List[PlayerWinningsCollected]:
    money_collected = []
    for pid, payout in payouts.items():
        money_collected.append(PlayerWinningsCollected(player_name=POSITIONS[pid],
                                                       collected="$" + str(payout),
                                                       rake=None))
    return money_collected


def _parse_cards(cards):
    # In: '3h, 9d, '
    # Out: '[Ah Jd]'
    tokens = cards.split(',')  # ['3h', ' 9d', ' ']
    c0 = tokens[0].replace(' ', '')
    c1 = tokens[1].replace(' ', '')
    return f'[{c0},  {c1}]'


def _get_remaining_players(env) -> List[PlayerWithCards]:
    # make player cards
    remaining_players = []
    for i in range(env.env.N_SEATS):
        # If player did not fold and still has money (is active and does not sit out)
        # append it to the remaining players
        if not env.env.seats[i].folded_this_episode:
            if env.env.seats[i].stack > 0 or env.env.seats[i].is_allin:
                cards = env.env.cards2str(env.env.get_hole_cards_of_player(i))
                remaining_players.append(PlayerWithCards(name=POSITIONS[i],
                                                         cards=_parse_cards(cards)))
    return remaining_players


@gin.configurable
def evaluate_baseline(path_to_torch_model_state_dict,
                      n_episodes,
                      test_gin_register=None
                      ):
    # global info  # convenient access to episode summary metrics
    starting_stacks = [1000, 1000]
    num_players = len(starting_stacks)
    env = create_wrapped_environment(starting_stacks)

    agents = make_agents(env, path_to_torch_model_state_dict)
    assert len(agents) == num_players

    total_actions_dict = {0: 0, 1: 0, 2: 0}
    for ep_id in range(n_episodes):
        # -------- Reset environment ------------
        obs, _, done, _ = env.reset()
        showdown_hands = None
        normalization_sum = float(
            sum([s.starting_stack_this_episode for s in env.env.seats])
        ) / env.env.N_SEATS
        blinds = _get_blinds(obs, num_players, normalization_sum)
        ante = obs[COLS.Ante] * normalization_sum
        player_stacks = _get_player_stacks(obs, num_players, normalization_sum)
        actions_total = {'preflop': [],
                         'flop': [],
                         'turn': [],
                         'river': []}
        # make obs
        legal_moves = env.env.get_legal_actions()
        observation = {'obs': [obs], 'legal_moves': [legal_moves]}
        agent_idx = 0

        while not done:
            # -------- COMPUTE ACTION -----------
            action_vec = agents[agent_idx].act(observation)
            # Record action to episodes actions
            action = int(action_vec[0][0].numpy())
            agent_that_acted = POSITIONS[-int(obs[COLS.Btn_idx])]

            stage = Poker.INT2STRING_ROUND[env.env.current_round]
            action_type = ACTION_TYPES[min(action, 2)]
            raise_amount = env.env.last_action[1]
            episode_action = Action(stage=stage,
                                    player_name=agent_that_acted,
                                    action_type=action_type,
                                    raise_amount=raise_amount)
            actions_total[stage].append(episode_action)
            remaining_players = _get_remaining_players(env)

            # -------- STEP ENVIRONMENT -----------
            obs, _, done, info = env.step(action)
            # if not done, prepare next turn
            if done:
                showdown_hands = remaining_players
                print(ep_id)
            legal_moves = env.env.get_legal_actions()
            observation = {'obs': [obs], 'legal_moves': [legal_moves]}

            # -------- SET NEXT AGENT -----------
            agent_idx += 1
            agent_idx = agent_idx % num_players

            # env.env.cards2str(env.env.get_hole_cards_of_player(0))
        a = "debug"
        winners = _get_winners(showdown_players=showdown_hands, payouts=info['payouts'])
        money_collected = _get_money_collected(payouts=info['payouts'])
        episode = PokerEpisode(date=str(datetime.now()),
                               hand_id=ep_id,
                               variant="HUNL",
                               currency_symbol="$",
                               num_players=num_players,
                               blinds=blinds,
                               ante=ante,
                               player_stacks=player_stacks,
                               btn_idx=0,
                               board_cards=env.env.cards2str(env.env.board),
                               actions_total=actions_total,
                               winners=winners,
                               showdown_hands=showdown_hands,
                               money_collected=money_collected)
        debug = 'set breakpoint here'
    print(total_actions_dict)

    # todo: make this the sanity check if very tight agent performs better vs calling station
    #  and the second ipynb should evaluate the agent purely in self play


if __name__ == '__main__':
    # configure run
    gin.config.external_configurable(Dummy)
    gin.parse_config_file(os.environ['PRL_BASELINES_ROOT_DIR'] + '/config.gin')
    evaluate_baseline()

    # have some plots at the end too
    # run games, collect metrics, parse to PokerEpisode
    # PokerEpisode into PokerSnowie
