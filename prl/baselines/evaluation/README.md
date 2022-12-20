# prl.baselines.eval
Experiment class `experiment.PokerExperiment`
provides configurable experiments.

`runner.run(PokerExperiment)` runs an experiment
and returns a list of Poker episodes `List[PokerEpisode]`
for evaluation. The result of an evaluation is represented 
by an `Evaluation` class instance, which are returned from
`evaluator.evaluate(List[PokerEpisodes], eval_config)`