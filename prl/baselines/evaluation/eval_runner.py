# one game predefined agent
from prl.baselines.agents.dummy_agents import DummyAgentFold, DummyAgentCall, DummyAgentAllIn
from prl.baselines.evaluation.core.experiment import make_participants, PokerExperiment, make_default_experiment
from prl.baselines.evaluation.utils import get_default_env, print_player_stacks

n_players = 4
starting_stack_size = 20000
env = get_default_env(n_players, starting_stack_size=starting_stack_size)

agent_fold = DummyAgentFold()
agent_call = DummyAgentCall()
agent_allin = DummyAgentAllIn()
agents = [agent_call for i in range(n_players)]
# make experiment with reset config that has cards so we know exactly who wins
env_reset_config = {}
participants = make_participants(agents, starting_stack_size)
experiment = make_default_experiment(env, participants, env_reset_config=env_reset_config)
# let the corresponding agents move and compare the text outputs

