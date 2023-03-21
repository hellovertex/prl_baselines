import json
import glob
from pathlib import Path

files = glob.glob('/home/sascha/Documents/github.com/prl_baselines/data/99_summary'
                  '/player_stats' + '**/*.json', recursive=False)

# todo using mutliprocessing on player names
# rewrite _select_hands(...) such that it
# counts hands where player has folded
# count hands where player reached showdown even if mucked
# how are we gonna compute stats when we dont have observations
# todo with the goal of re-running eval_analyzer.py - like evaluation of selected players
#  but this time with the correct numbers
# this will only be done once so it does not have to have a perfect api
# todo then compute baseline stats from 1M games vs random agents
best_players = ['ishuha',
                'Sakhacop',
                'nastja336',
                'Lucastitos',
                'I LOVE RUS34',
                'SerAlGog', ]
# 'Ma1n1',
# 'zMukeha',
# 'SoLongRain',
# 'LuckyJO777',
# 'Nepkin1',
# 'blistein',
# 'ArcticBearDK',
# 'Creator_haze',
# 'ilaviiitech',
# 'm0bba',
# 'KDV707']
worst_players = ['Calabazaking',
                 'jenya86rus',
                 'podumci',
                 'Inas21',
                 'Fischkop993']

dicts_good = {}
dicts_bad = {}
d={}
for f in files:
    pname = Path(f).stem.split('stats_')[1]
    d = json.load(open(f, mode='r'))

    if pname in best_players:
        dicts_good[pname] = d
    else:
        dicts_bad[pname] = d
res = ''
# for k,v in dicts_good.items():
#     res += k
#     for stat, val in v.items():
#         if stat in ['vpip', 'pfr', 'af']:
#             res += f'& {round(val,2)}'
#     n_showdowns = round(v['total_number_of_samples'] * v['wtsd'])
#     sklansky = 'Loose' if v['vpip'] > .25 else 'Tight'
#     sklansky += ' Passive' if v['af'] < 1 else ' Aggressive'
#     res += f'& {n_showdowns} & {v["won"]} & {sklansky}'
#     res += '\n'
# print(res)
res = ''
for k,v in dicts_bad.items():
    res += k
    for stat, val in v.items():
        if stat in ['vpip', 'pfr', 'af']:
            res += f'& {round(val,2)}'
    n_showdowns = round(v['total_number_of_samples'] * v['wtsd'])
    sklansky = 'Loose' if v['vpip'] > .25 else 'Tight'
    sklansky += ' Passive' if v['af'] < 1 else ' Aggressive'
    res += f'& {n_showdowns} & {v["won"]} & {sklansky}'
    res += '\n'
print(res)
a = 1
from faker import Faker
fake = Faker()

for i in range(0, 10):
    print(fake.name())