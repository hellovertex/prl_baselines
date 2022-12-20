from typing import Dict, Any, List

import numpy as np
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env
from prl.environment.steinberger.PokerRL import Poker

from prl.baselines.agents.agents import BaselineAgent
from prl.baselines.evaluation.core.evaluator import DEFAULT_CURRENCY, DEFAULT_DATE, DEFAULT_VARIANT
from prl.baselines.evaluation.core.experiment import PokerExperiment
from prl.baselines.evaluation.experiment_runner import PokerExperimentRunner
from prl.baselines.agents.eval.utils import make_participants
from prl.baselines.agents.policies import CallingStation
from prl.baselines.supervised_learning.data_acquisition.core.encoder import PlayerInfo, Positions6Max
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode, Blind, PlayerStack, ActionType, \
    Action, PlayerWithCards, PlayerWinningsCollected
from prl.baselines.supervised_learning.data_acquisition.environment_utils import card_tokens, card
from prl.baselines.supervised_learning.data_acquisition.rl_state_encoder import RLStateEncoder


def make_state_dict(player_hands: List[str], board_cards: str) -> Dict[str, Any]:
    hands = []
    for hand in player_hands:
        hands.append([card(token) for token in card_tokens(hand)])
    board = [card(token) for token in card_tokens(board_cards)]
    deck = np.empty(shape=(13 * 4, 2), dtype=np.int8)
    initial_board = np.full((5, 2), Poker.CARD_NOT_DEALT_TOKEN_1D, dtype=np.int8)
    deck[:len(board)] = board
    return {'deck': {'deck_remaining': deck},  # np.ndarray(shape=(52-n_cards*num_players, 2))
            'board': initial_board,  # np.ndarray(shape=(n_cards, 2))
            'hand': hands}


def test_episode_matches_environment_states_and_actions():
    # todo rename to test_PokerExperimentRunner
    # setup
    encoder = RLStateEncoder()  # to parse human-readable cards to 2d arrays
    num_players = 2
    starting_stack_size = 1000
    table = tuple([PlayerInfo(seat_number=i,
                              position_index=i,
                              position=Positions6Max(i).name,
                              player_name=f'Player_{i}',
                              stack_size=starting_stack_size) for i in range(num_players)])
    # 1. need deck_state_dict = hand cards + board
    hand_0 = '[Ah Kh]'
    hand_1 = '[7s 2c]'
    player_hands = [hand_0, hand_1]
    board = 'Qh Jh Th 9h 8h'
    env_config = {'deck_state_dict': make_state_dict(player_hands, board)}
    test_env = init_wrapped_env(AugmentObservationWrapper,
                                [starting_stack_size for _ in range(num_players)],
                                blinds=[50, 100],
                                multiply_by=1)
    # 2. need action sequence that results in showdown (any will do, e.g. all in and call)
    actions = [(2, 500),  # p0 bets his AhKh
               (1, -1),  # p1 calls with his crap hand
               (1, -1),  # p0 checks on flop
               (1, -1),  # p1 checks on flop
               (1, -1),  # p1 checks on turn
               (1, -1),  # p1 checks on turn
               (1, -1),  # p1 checks on river
               (1, -1),  # p1 checks on river
               ]
    action_list = [Action('preflop', 'Player_0', ActionType.RAISE, 500.0),
                   Action('preflop', 'Player_1', ActionType.CHECK_CALL, 500.0),
                   Action('flop', 'Player_1', ActionType.CHECK_CALL, 0.0),
                   Action('flop', 'Player_0', ActionType.CHECK_CALL, 0.0),
                   Action('turn', 'Player_1', ActionType.CHECK_CALL, 0.0),
                   Action('turn', 'Player_0', ActionType.CHECK_CALL, 0.0),
                   Action('river', 'Player_1', ActionType.CHECK_CALL, 0.0),
                   Action('river', 'Player_0', ActionType.CHECK_CALL, 0.0)]
    actions_total = {'preflop': [Action('preflop', 'Player_0', ActionType.RAISE, 500.0),
                                 Action('preflop', 'Player_1', ActionType.CHECK_CALL, 500.0)],
                     'flop': [Action('flop', 'Player_1', ActionType.CHECK_CALL, 0.0),
                              Action('flop', 'Player_0', ActionType.CHECK_CALL, 0.0)],
                     'turn': [Action('turn', 'Player_1', ActionType.CHECK_CALL, 0.0),
                              Action('turn', 'Player_0', ActionType.CHECK_CALL, 0.0)],
                     'river': [Action('river', 'Player_1', ActionType.CHECK_CALL, 0.0),
                               Action('river', 'Player_0', ActionType.CHECK_CALL, 0.0)],
                     'as_sequence': action_list}
    # 3. construct PokerEpisode that we expect
    expected_episode = PokerEpisode(date=DEFAULT_DATE,
                                    hand_id=0,
                                    variant=DEFAULT_VARIANT,
                                    currency_symbol=DEFAULT_CURRENCY,
                                    num_players=num_players,
                                    blinds=[Blind("Player_0", 'small blind', '$50'),
                                            Blind("Player_1", 'big blind', '$100')],
                                    ante='$0.00',
                                    player_stacks=[PlayerStack('Seat 0', 'Player_0', '$950'),
                                                   PlayerStack('Seat 1', 'Player_1', '$900')],
                                    btn_idx=0,
                                    board_cards=f'[{board}]',
                                    actions_total=actions_total,
                                    winners=[PlayerWithCards('Player_0', f'{hand_0}')],
                                    showdown_hands=[PlayerWithCards('Player_0', f'{hand_0}'),
                                                    PlayerWithCards('Player_1', f'{hand_1}')],
                                    money_collected=[PlayerWinningsCollected('Player_0', "$1000.0", None)]
                                    )
    experiment = PokerExperiment(num_players=num_players,
                                 env=test_env,
                                 env_config=env_config,
                                 participants={},  # no agents since we run from action list
                                 max_episodes=1,
                                 current_episode=0,
                                 cbs_plots=[],
                                 cbs_misc=[],
                                 cbs_metrics=[],
                                 from_action_plan=actions
                                 )
    runner = PokerExperimentRunner()
    returned_episodes = runner.run(experiment)
    ret_dict = returned_episodes[0]._asdict()
    for k, v in expected_episode._asdict().items():
        print(f'k: {k}, \nv: {v}, \n'
              f'vx: {ret_dict[k]}')

    assert expected_episode == returned_episodes[0]


def test_blinds_alternate():
    num_players = 2
    starting_stack_size = 1000
    test_env = init_wrapped_env(AugmentObservationWrapper,
                                [starting_stack_size for _ in range(num_players)],
                                blinds=[50, 100],
                                multiply_by=1)
    reference_policy = CallingStation(test_env.observation_space, test_env.action_space, {})
    reference_agent = BaselineAgent({'rllib_policy': reference_policy})
    agents = [reference_agent, reference_agent]
    experiment = PokerExperiment(num_players=num_players,
                                 env=test_env,
                                 env_config=None,
                                 participants=make_participants(agents, starting_stack_size),
                                 max_episodes=5,
                                 current_episode=0,
                                 cbs_plots=[],
                                 cbs_misc=[],
                                 cbs_metrics=[],
                                 from_action_plan=None
                                 )
    runner = PokerExperimentRunner()
    returned_episodes = runner.run(experiment)
    # make sure blinds are alternating
    assert returned_episodes[0].blinds[0].player_name == 'Player_0'
    assert returned_episodes[1].blinds[0].player_name == 'Player_1'
    assert returned_episodes[2].blinds[0].player_name == 'Player_0'
    assert returned_episodes[3].blinds[0].player_name == 'Player_1'
    assert returned_episodes[4].blinds[0].player_name == 'Player_0'

def test_player_stacks():
    pass