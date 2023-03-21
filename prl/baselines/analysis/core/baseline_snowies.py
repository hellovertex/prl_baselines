from prl.baselines.agents.tianshou_agents import ImitatorAgent
from prl.baselines.analysis.core.stats import PlayerStats
from prl.baselines.evaluation.core.experiment import PokerExperiment, make_participants
from prl.baselines.evaluation.pokersnowie.export import PokerExperimentToPokerSnowie
from prl.baselines.evaluation.v2.eval_agent import EvalAgentTianshou, \
    EvalAgentRandom
from prl.baselines.examples.examples_tianshou_env import make_default_tianshou_env


def main(max_episodes, agents, agent_names, pname):
    assert len(agents) == len(agent_names)
    num_players = len(agents)
    path_out = "./baseline_stats"
    starting_stack = 5000
    env = make_default_tianshou_env(mc_model_ckpt_path=None,  # dont use mc
                                    stack_sizes=[starting_stack for _ in
                                                 range(len(agent_names))],
                                    agents=agent_names,
                                    num_players=len(agent_names))
    normalization = env.env.env.env_wrapped.normalization
    assert len(agents) == num_players
    participants = make_participants(agents=agents,
                                     agent_names=agent_names,
                                     starting_stack=starting_stack,
                                     normalization=normalization)
    stats = [PlayerStats(pname=pname) for pname in agent_names]
    experiment = PokerExperiment(
        wrapped_env=env,  # single environment to run sequential games on
        num_players=num_players,
        starting_stack_sizes=[starting_stack for _ in range(num_players)],
        env_reset_config=None,
        max_episodes=max_episodes,  # number of games to run
        current_episode=0,
        cbs_plots=[],
        cbs_misc=[],
        cbs_metrics=[],
        options={'stats': stats},
        participants=participants,
        from_action_plan=None
    )
    PokerExperimentToPokerSnowie().generate_database(
        verbose=True,
        path_out=path_out + f'/{pname}',
        experiment=experiment,
        max_episodes_per_file=10,
        hero_names=[pname]
    )


def load_imitation_agent(ckpt_dir):
    return ImitatorAgent(ckpt_dir,
                         flatten_input=False,
                         model_hidden_dims=(512,))


if __name__ == '__main__':
    max_episodes = 10
    num_players = 6
    pname = 'AI_AGENT_v2'
    agent_names = [f'{pname}', '2', '3', '4', '5', '6']
    ckpt_dir = '/home/sascha/Documents/github.com/prl_baselines/data/05_train_results/from_gdrive/05_train_results-20230320T232447Z-001/05_train_results/NL50/player_pool/folds_from_top_players_with_randomized_hand/Top20Players_n_showdowns=5000/target_rounds=FTR/actions=ActionSpaceMinimal/512_1e-06/ckptdir/ckpt.pt'
    # if you want to load RL learners, use the loading_fns from train_eval.py
    agent = load_imitation_agent(ckpt_dir)
    agents = [EvalAgentTianshou(pname, agent)]
    agents += [EvalAgentRandom(f'p{i + 1}') for i in
               range(num_players - 1)]
    main(max_episodes, agents, agent_names, pname)
