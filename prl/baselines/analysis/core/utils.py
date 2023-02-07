from prl.baselines.agents.tianshou_agents import BaselineAgent
from prl.baselines.analysis.core.stats import PlayerStats
from prl.baselines.evaluation.core.experiment import PokerExperiment, make_participants
from prl.baselines.examples.examples_tianshou_env import make_default_tianshou_env


def make_experiment(max_episodes,
                    num_players,
                    agent_names,
                    ckpt_abs_fpath,
                    hidden_dims):
    starting_stack = 20000
    env = make_default_tianshou_env(mc_model_ckpt_path=None,  # dont use mc
                                    agents=agent_names,
                                    num_players=len(agent_names))

    # make self play agents
    agents = [BaselineAgent(ckpt_abs_fpath,  # MajorityBaseline
                            flatten_input=False,
                            num_players=num_players,
                            model_hidden_dims=hidden_dims) for _ in range(num_players)]  # make self play agents

    assert len(agents) == num_players
    participants = make_participants(agents, starting_stack)
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
        options={'stats': stats},
        # actors
        participants=participants,  # wrapper around agents that hold rllib policies that act given observation
        from_action_plan=None,  # compute action from fixed series of actions instead of calls to agent.act
        # early_stopping_when=PokerExperiment_EarlyStopping.ALWAYS_REBUY_AND_PLAY_UNTIL_NUM_EPISODES_REACHED
    )
    return experiment
