import os

import pandas as pd
import json

def json_to_df(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path) as f:
            d =json.load(f)
            cbet = d.pop('cbet')
            d['cbet_flop'] = cbet['flop']
            d['cbet_turn'] = cbet['turn']
            d['cbet_river'] = cbet['river']
            data.append(d)
    return pd.DataFrame(data)

def get_json_files(directory):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files

if __name__ == "__main__":
    json_file_list = get_json_files("/home/sascha/Documents/github.com/prl_baselines/data/stats_baseline_nets/pokersnowie")
    df = json_to_df(json_file_list)
    print(df.head())
    df.to_csv('json_stats_player_summary.csv')
