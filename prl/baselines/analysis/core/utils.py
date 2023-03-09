from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env

from prl.baselines.agents.tianshou_agents import BaselineAgent
from prl.baselines.analysis.core.stats import PlayerStats
from prl.baselines.evaluation.core.experiment import PokerExperiment, make_participants
from prl.baselines.examples.examples_tianshou_env import make_default_tianshou_env


def make_experiment(max_episodes,
                    num_players,
                    agent_names,
                    agents,
                    ckpt_abs_fpath,
                    hidden_dims):
    starting_stack = 5000
    env = make_default_tianshou_env(mc_model_ckpt_path=None,  # dont use mc
                                    stack_sizes=[starting_stack for _ in range(len(agent_names))],
                                    agents=agent_names,
                                    num_players=len(agent_names))
    normalization = env.env.env.env_wrapped.normalization
    test_env = init_wrapped_env(AugmentObservationWrapper,
                                [starting_stack for _ in range(len(agent_names))],
                                blinds=(25, 50),
                                multiply_by=1)

    assert len(agents) == num_players
    participants = make_participants(agents=agents,
                                     agent_names=agent_names,
                                     starting_stack=starting_stack,
                                     normalization=normalization)
    stats = [PlayerStats(pname=pname) for pname in agent_names]
    # run self play
    experiment = PokerExperiment(
        # env
        wrapped_env=env,  # single environment to run sequential games on
        num_players=num_players,
        starting_stack_sizes=[starting_stack for _ in range(num_players)],
        env_reset_config=None,  # can pass {'deck_state_dict': Dict[str, Any]} to init the deck and player cards
        # run
        max_episodes=max_episodes,  # number of games to run
        current_episode=0,
        cbs_plots=[],
        cbs_misc=[],
        cbs_metrics=[],
        options={'stats': stats,
                 'test_env': test_env},
        # actors
        participants=participants,  # wrapper around agents that hold rllib policies that act given observation
        from_action_plan=None,  # compute action from fixed series of actions instead of calls to agent.act
        # early_stopping_when=PokerExperiment_EarlyStopping.ALWAYS_REBUY_AND_PLAY_UNTIL_NUM_EPISODES_REACHED
    )
    return experiment
