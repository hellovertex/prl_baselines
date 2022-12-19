from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env

from prl.baselines.agents.eval.core.experiment import PokerExperiment
from prl.baselines.agents.eval.experiment_runner import PokerExperimentRunner
from prl.baselines.agents.eval.utils import make_agents, make_participants
from prl.baselines.pokersnowie.exporteur import PokerExperimentToPokerSnowie

if __name__ == '__main__':
    # move this to example.py or main.py
    # Construct Experiment
    starting_stack_size = 1000
    num_players = 2
    stacks = [starting_stack_size for _ in range(num_players)]
    env_wrapped = init_wrapped_env(env_wrapper_cls=AugmentObservationWrapper,
                                   stack_sizes=stacks,
                                   multiply_by=1)
    max_episodes = 10
    path_to_baseline_torch_model_state_dict = "/home/sascha/Documents/github.com/prl_baselines/data/ckpt.pt"
    agent_list = make_agents(env_wrapped, path_to_baseline_torch_model_state_dict)
    participants = make_participants(agent_list, starting_stack_size)
    experiment = PokerExperiment(
        env=env_wrapped,
        num_players=num_players,
        env_config=None,
        participants=participants,
        max_episodes=max_episodes,
        current_episode=0,
        cbs_plots=[],
        cbs_misc=[],
        cbs_metrics=[],
        from_action_plan=None
    )
    db_gen = PokerExperimentToPokerSnowie()
    db_gen.generate_database(path_out='.',
                             experiment=experiment,
                             max_episodes_per_file=500)

