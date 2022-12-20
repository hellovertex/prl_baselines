from typing import Dict, Any, List

import numpy as np
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env
from prl.environment.steinberger.PokerRL import Poker

from prl.baselines.agents.agents import BaselineAgent
from prl.baselines.evaluation.core.experiment import PokerExperiment, AGENT, PokerExperimentParticipant, DEFAULT_DATE, \
    DEFAULT_VARIANT, DEFAULT_CURRENCY
from prl.baselines.evaluation.experiment_runner import PokerExperimentRunner
from prl.baselines.agents.policies import CallingStation, StakeLevelImitationPolicy
from prl.baselines.supervised_learning.data_acquisition.core.encoder import PlayerInfo, Positions6Max
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode, Blind, PlayerStack, ActionType, \
    Action, PlayerWithCards, PlayerWinningsCollected
from prl.baselines.supervised_learning.data_acquisition.environment_utils import card_tokens, card
from prl.baselines.supervised_learning.data_acquisition.rl_state_encoder import RLStateEncoder


def make_agents(env, path_to_torch_model_state_dict):
    policy_config = {'path_to_torch_model_state_dict': path_to_torch_model_state_dict}
    baseline_policy = StakeLevelImitationPolicy(env.observation_space, env.action_space, policy_config)
    reference_policy = CallingStation(env.observation_space, env.action_space, policy_config)

    baseline_agent = BaselineAgent({'rllib_policy': baseline_policy})
    reference_agent = BaselineAgent({'rllib_policy': reference_policy})
    return [baseline_agent, baseline_agent]


def make_participants(agents: List[AGENT], starting_stack_size: int) -> Dict[int, PokerExperimentParticipant]:
    participants = {}
    for i, agent in enumerate(agents):
        participants[i] = PokerExperimentParticipant(id=i,
                                                     name=f'{type(agent.policy)}',
                                                     alias=f'Agent_{i}',
                                                     starting_stack=starting_stack_size,
                                                     agent=agent,
                                                     config={})
    return participants


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


DEFAULT_STARTING_STACK_SIZE = 1000


def test_three_players():
    num_players = 3
    deck_state_dict = make_state_dict(
        player_hands=[
            '[7s 2c]',
            '[Ah Kh]',
            '[Qd Qc]'
        ],
        board_cards='Qh Jh Th 9h 8h')
    env_config = {'deck_state_dict': deck_state_dict}
    test_env = init_wrapped_env(AugmentObservationWrapper,
                                [DEFAULT_STARTING_STACK_SIZE for _ in range(num_players)],
                                blinds=[50, 100],
                                multiply_by=1)


def get_actions_total(actions: List[Action]) -> Dict[str, List[Action]]:
    actions_total = {'preflop': [],
                     'flop': [],
                     'turn': [],
                     'river': []}
    for a in actions:
        actions_total[a.stage].append(a)
    return actions_total


def test_episode_matches_environment_states_and_actions():
    num_players = 2
    starting_stack_size = 1000
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
    action_list = [Action('preflop', 'Player_0', ActionType.RAISE, 500),
                   Action('preflop', 'Player_1', ActionType.CHECK_CALL, 500),
                   Action('flop', 'Player_1', ActionType.CHECK_CALL, 0),
                   Action('flop', 'Player_0', ActionType.CHECK_CALL, 0),
                   Action('turn', 'Player_1', ActionType.CHECK_CALL, 0),
                   Action('turn', 'Player_0', ActionType.CHECK_CALL, 0),
                   Action('river', 'Player_1', ActionType.CHECK_CALL, 0),
                   Action('river', 'Player_0', ActionType.CHECK_CALL, 0)]
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
                                    actions_total=get_actions_total(action_list),
                                    winners=[PlayerWithCards('Player_0', f'{hand_0}')],
                                    showdown_hands=[PlayerWithCards('Player_0', f'{hand_0}'),
                                                    PlayerWithCards('Player_1', f'{hand_1}')],
                                    money_collected=[PlayerWinningsCollected('Player_0', "$1000", None)]
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
                                 from_action_plan=[action_list]
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
