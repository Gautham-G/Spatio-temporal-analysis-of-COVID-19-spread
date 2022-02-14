import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import optparse
from datetime import datetime
from glob import glob
import json, ast
from tqdm import tqdm
import pickle

start_week = '2020-08-17'
curr_week = '2021-09-27'
ahead = 1

def week_to_number(week: str):
    start_date = datetime.strptime(start_week, '%Y-%m-%d')
    curr_date = datetime.strptime(week, '%Y-%m-%d')
    return (curr_date - start_date).days // 7

curr_week_num = week_to_number(curr_week)

# Get zip time-series data
df = pd.read_csv('data/caserate_by_zcta_cleaned.csv')
df["week_no"] = df["week_ending"].apply(lambda x: week_to_number(x))

# Make date dict
date_dict = {}
ct = 0
for i in df["ZIP"].unique():
    date_dict[i] = ct
    ct += 1

# Make timeseries arrays
tseries = np.zeros((len(df["ZIP"].unique()), curr_week_num + ahead))
for i in tqdm(df["ZIP"].unique()):
    for j in range(45):
        tseries[date_dict[i], j] = df["caseRate"][(df["ZIP"] == i) & (df["week_no"] == j)].values[0]

os.makedirs('./saves', exist_ok=True)
with open('./saves/tseries.pkl', 'wb') as f:
    pickle.dump(tseries, f)

# Get visit counts
pattern_files = glob("data/patterns_cleaned/*.csv")
df1 = pd.read_csv(pattern_files[0])
for f in pattern_files[1:]:
    df1 = df1.append(pd.read_csv(f))
df1['week_no'] = df1['date_range_end'].apply(lambda x: week_to_number(x))

def get_tot_cts(dic):
    return sum(ast.literal_eval(dic).values())

df1['tot_visits'] = df1['visitor'].apply(lambda x: get_tot_cts(x))

with open('./saves/visit_counts_df.pkl', 'wb') as f:
    pickle.dump(df1, f)


visits_count = []
for w in range(45):
    print(f"Week {w}")
    df_w = df1[(df1['week_no'] == w)][['placekey','visitor']]
    v_dict = {}
    for p in tqdm(df_w['placekey'].unique()):
        v_dict[p] = ast.literal_eval(df_w[df_w['placekey'] == p]['visitor'].values[0])
    visits_count.append(v_dict)

with open('./saves/visits_count.pkl', 'wb') as f:
    pickle.dump(visits_count, f)

intersect_pois = set(visits_count[0].keys())
for i in range(1, len(visits_count)):
    intersect_pois = intersect_pois.intersection(set(visits_count[i].keys()))

poi_dict = {}
ct = 0
for i in intersect_pois:
    poi_dict[i] = ct
    ct += 1

visit_matrix = np.zeros((len(visits_count), len(poi_dict), len(date_dict)))
for i in range(len(visits_count)):
    for j in poi_dict.keys():
        for k in date_dict.keys():
            if str(k) in visits_count[i][j].keys():
                visit_matrix[i, poi_dict[j], date_dict[k]] = visits_count[i][j][str(k)] 


with open("./saves/visit_matrix.pkl", "wb") as f:
    pickle.dump(visit_matrix, f)