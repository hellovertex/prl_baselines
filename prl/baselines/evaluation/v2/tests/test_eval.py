from prl.baselines.agents.dummy_agents import DummyAgentCall, DummyAgentFold
from prl.baselines.evaluation.v2.eval import Evaluation
from prl.baselines.evaluation.v2.eval_agent import EvalAgent

def test_stats_are_correct():
    """Calling Station vs AlwaysFold"""
    calling_station = EvalAgent(name='callingstation',agent=DummyAgentCall())
    always_fold = EvalAgent(name='foldingagent',agent=DummyAgentFold())
    agents = [calling_station, always_fold]
    agent_ids_to_inspect = [0]

    # run single evaluation episode
    eval = Evaluation(record_stats=True)
    eval.agents = agents
    eval.run(n_episodes=1, eval_agent_target_indices=agent_ids_to_inspect)
    # assert vpip is 1 for caller and 0 for always fold
    # run multiple even number of rounds and assert .5 and 0

    eval.player_stats
