# one game predefined agent
from prl.baselines.agents.dummy_agents import DummyAgentFold, DummyAgentCall, DummyAgentAllIn
from prl.baselines.evaluation.core.experiment import make_participants, PokerExperiment, make_default_experiment
from prl.baselines.evaluation.experiment_runner import PokerExperimentRunner

from prl.baselines.evaluation.utils import get_default_env, print_player_stacks, get_reset_config


n_players = 4
starting_stack_size = 20000
env = get_default_env(n_players, starting_stack_size=starting_stack_size)

agent_fold = DummyAgentFold()
agent_call = DummyAgentCall()
agent_allin = DummyAgentAllIn()
agents = [agent_call for i in range(n_players)]
# make experiment with reset config that has cards so we know exactly who wins
board = '[6h Ts Td 9c Jc]'
player_hands = ['[6s 6d]', '[9s 9d]', '[Jd Js]', '[Th Tc]']
env_reset_config = get_reset_config(player_hands, board)

participants = make_participants(agents, starting_stack_size)
experiment = make_default_experiment(env,
                                     participants,
                                     max_episodes=1,
                                     env_reset_config=env_reset_config)
# let the corresponding agents move and compare the text outputs
runner = PokerExperimentRunner()
poker_episodes = runner.run(experiment, verbose=True)
print(poker_episodes)
