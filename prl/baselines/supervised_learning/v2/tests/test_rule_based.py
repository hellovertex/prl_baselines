from functools import partial

import pytest
from prl.environment.Wrappers.aoh import Positions6Max
from prl.environment.Wrappers.base import ActionSpace
from tianshou.env import PettingZooEnv

from prl.baselines.agents.rule_based import RuleBasedAgent
from prl.baselines.evaluation.utils import get_reset_config, pretty_print
from prl.baselines.examples.examples_tianshou_env import make_default_tianshou_env


def make_agent_names(num_players):
    return [f'p{i}' for i in range(num_players)]


@pytest.fixture
def env_six_players():
    num_players = 6
    starting_stack = 5000
    stack_sizes = [starting_stack for _ in range(num_players)]
    agent_names = make_agent_names(num_players)
    env = make_default_tianshou_env(mc_model_ckpt_path=None,  # dont use mc
                                    agents=agent_names,
                                    num_players=len(agent_names))
    assert len(agent_names) == num_players == len(stack_sizes)
    return env


def act(obs, agents, agent_names):
    agent_id = obs['agent_id']
    i = agent_names.index(agent_id)
    action = agents[i].act(obs['obs'], obs['mask'])
    amt = -1
    if isinstance(action, tuple):
        action, amt = action[0], action[1]
    action = min(action, 2)
    return action, amt


def test_btn_idx_moves_correctly(env_six_players):
    # arrange
    env = env_six_players
    num_players = env_six_players.num_agents
    normalization = env_six_players.env.env.env_wrapped.normalization
    agents = [RuleBasedAgent(num_players, normalization) for _ in range(num_players)]
    rule_based_agent = agents[0]
    agent_names = make_agent_names(num_players)
    obs_dict = env_six_players.reset()
    for pos in [
        # preflop
        Positions6Max.UTG,
        Positions6Max.MP,
        Positions6Max.CO,
        Positions6Max.BTN,
        Positions6Max.SB,
        Positions6Max.BB,
        # flop
        Positions6Max.SB,
        Positions6Max.BB,
        Positions6Max.UTG,
        Positions6Max.MP,
        Positions6Max.CO,
        Positions6Max.BTN,
        # turn
        Positions6Max.SB,
        Positions6Max.BB,
        Positions6Max.UTG,
        Positions6Max.MP,
        Positions6Max.CO,
        Positions6Max.BTN,
        # river
        Positions6Max.SB,
        Positions6Max.BB,
        Positions6Max.UTG,
        Positions6Max.MP,
        Positions6Max.CO,
        Positions6Max.BTN,
    ]:
        assert rule_based_agent.get_position_idx(obs_dict['obs']) == pos
        obs_dict, _, _, _, _ = env.step(ActionSpace.CHECK_CALL)


def test_is_first_betting_round(env_six_players):
    # init
    env = env_six_players
    num_players = env_six_players.num_agents
    normalization = env_six_players.env.env.env_wrapped.normalization
    agents = [RuleBasedAgent(num_players, normalization) for _ in range(num_players)]
    rule_based_agent = agents[0]
    for _ in [
        Positions6Max.UTG,
        Positions6Max.MP,
        Positions6Max.CO,
        Positions6Max.BTN,
        Positions6Max.SB,
        Positions6Max.BB
    ]:
        obs_dict, _, _, _, _ = env.step(ActionSpace.RAISE_MIN_OR_THIRD_OF_POT)
        raises = rule_based_agent.get_raises_preflop(obs_dict['obs'])
        assert rule_based_agent.is_first_betting_round(raises)
    obs_dict, _, _, _, _ = env.step(ActionSpace.RAISE_MIN_OR_THIRD_OF_POT)
    raises = rule_based_agent.get_raises_preflop(obs_dict['obs'])
    assert not rule_based_agent.is_first_betting_round(raises)


def test_rule_based_six(env_six_players):
    # arrange
    env = env_six_players
    num_players = env_six_players.num_agents
    normalization = env_six_players.env.env.env_wrapped.normalization
    agents = [RuleBasedAgent(num_players, normalization) for _ in range(num_players)]
    agent_names = make_agent_names(num_players)
    act_current = partial(act, agents=agents, agent_names=agent_names)

    # init env
    # action_sequence = [2, 1, 2, 2, 1, 2, 2]
    hero_hand = 'J9o'
    # board = '[Ks Kh Kd Kc 2s]'
    board = '[4s 4h 4d 4c 2s]'
    player_hands = [
        '[Ah Ac]', '[Kh Ks]', '[As Ad]', '[Jd 9h]', 'Ts 5h', '8s Td'
    ]
    expected_preflop_actions = [
        # UTG, MP, CO FOLD
        # BB, SB, BB 3bet, 4bet, all in
        0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
    ]
    state_dict = get_reset_config(player_hands, board)
    options = {'reset_config': state_dict}

    # run
    obs_dict = env_six_players.reset(options=options)
    for a in expected_preflop_actions:
        action, amt = act_current(obs_dict)
        assert action == a
        obs_dict, cum_reward, terminated, truncated, info = env.step(action)
