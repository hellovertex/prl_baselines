# run after snowies
from prl.baselines.analysis.core.player_snowies import HSmithyStats


def main(filename, pname):
    stats = HSmithyStats(pname)
    stats.compute_from_file(
        filename,
        pname,
        hand_separator='#Game No :'
    )

    print(stats.pstats.to_dict())


if __name__ == '__main__':
    pname = 'AI_AGENT_v2'
    filename = '/home/sascha/Documents/github.com/prl_baselines/prl/baselines/analysis/core/baseline_stats/AI_AGENT_v2/snowie_0.txt'
