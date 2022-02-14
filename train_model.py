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
from model.encoders import FFN
import optparse

start_week = "2020-08-17"
curr_week = "2021-09-27"

# Command line arguments
parser = optparse.OptionParser()
parser.add_option('-a', '--ahead', dest="ahead", default=1, help="Weeks ahead")
parser.add_option('-e', '--epochs', dest="epochs", default=1000, help="Number of epochs")
parser.add_option('-l', '--lr', dest="lr", default=0.001, help="Learning rate")

(options, args) = parser.parse_args()
ahead = int(options.ahead)
epochs = int(options.epochs)
lr = float(options.lr)


device = "cuda" if th.cuda.is_available() else "cpu"


def week_to_number(week: str):
    start_date = datetime.strptime(start_week, "%Y-%m-%d")
    curr_date = datetime.strptime(week, "%Y-%m-%d")
    return (curr_date - start_date).days // 7


curr_week_num = 45

# Get zip time-series data
df = pd.read_csv("data/caserate_by_zcta_cleaned.csv")
df["week_no"] = df["week_ending"].apply(lambda x: week_to_number(x))

# Make date dict
date_dict = {}
ct = 0
for i in df["ZIP"].unique():
    date_dict[i] = ct
    ct += 1


os.makedirs("./saves", exist_ok=True)
with open("./saves/tseries.pkl", "rb") as f:
    tseries = pickle.load(f)

# Min max scaler
min_max_scaler = lambda x: (x - x.min()) / (x.max() - x.min())
tseries = (tseries - np.min(tseries, axis=1, keepdims=True)) / (np.max(tseries, axis=1, keepdims=True) - np.min(tseries, axis=1, keepdims=True))

with open("./saves/visit_counts_df.pkl", "rb") as f:
    df1 = pickle.load(f)

with open("./saves/visits_count.pkl", "rb") as f:
    visits_count = pickle.load(f)

intersect_pois = set(visits_count[0].keys())
for i in range(1, len(visits_count)):
    intersect_pois = intersect_pois.intersection(set(visits_count[i].keys()))

poi_dict = {}
ct = 0
for i in intersect_pois:
    poi_dict[i] = ct
    ct += 1


with open("./saves/visit_matrix.pkl", "rb") as f:
    visit_matrix = pickle.load(f)

visit_matrix = th.from_numpy(visit_matrix).float()


class ZipEncoder(nn.Module):
    def __init__(self, embed_sz, num_zip, final_sz, rnn_dim):
        super(ZipEncoder, self).__init__()
        self.embed_layer = nn.Embedding(num_zip, embed_sz)
        self.fc = FFN(embed_sz, [60, 60], final_sz)
        self.rnn = nn.GRU(1, rnn_dim, batch_first=True)
        self.fc2 = FFN(rnn_dim + final_sz, [60, 60], 1)

    def forward(self, zip_no, sequences):
        embed = self.embed_layer(zip_no)
        rnn_out, _ = self.rnn(sequences.unsqueeze(-1))
        rnn_out = rnn_out[:, -1, :]
        x = self.fc(embed)
        x = th.cat([x, rnn_out], dim=1)
        x = self.fc2(x)
        return x, embed


class PoiEncoder(nn.Module):
    def __init__(self, embed_sz, num_poi) -> None:
        super(PoiEncoder, self).__init__()
        self.embed_layer = nn.Embedding(num_poi, embed_sz)
        self.fc = FFN(embed_sz, [60, 60], 1)

    def forward(self, poi_no):
        embed = self.embed_layer(poi_no)
        x = self.fc(embed)
        return x, embed


class WeightsEncoder(nn.Module):
    def __init__(self, embed_sz, final_sz):
        super(WeightsEncoder, self).__init__()
        self.fc_zip = nn.Linear(embed_sz, final_sz)
        self.fc_poi = nn.Linear(embed_sz, final_sz)
        self.lamda = nn.Parameter(th.tensor(0.1).to(device), requires_grad=True)

    def forward(self, zip_embeds, poi_embeds, visit_matrix):
        zip_embeds = self.fc_zip(zip_embeds)
        poi_embeds = self.fc_poi(poi_embeds)
        product_embeds = zip_embeds @ th.transpose(poi_embeds, 0, 1)
        weights = th.exp(self.lamda) * th.sigmoid(product_embeds) * visit_matrix
        return weights

z_encoder = ZipEncoder(embed_sz=40, num_zip=len(date_dict), final_sz=50, rnn_dim=50).to(device)
#ez, em_z = z_encoder.forward(th.LongTensor([0,1,3]), th.randn(3, 10))

p_encoder = PoiEncoder(embed_sz=40, num_poi=len(poi_dict)).to(device)
#ep, em_p = p_encoder.forward(th.LongTensor([0,1,3,4,5]))

w_encoder = WeightsEncoder(embed_sz=40, final_sz=50).to(device)
#wt = w_encoder.forward(em_z, em_p, th.rand(3, 5))


train_series = tseries[:, :curr_week_num]

opt = th.optim.Adam(list(z_encoder.parameters()) + list(p_encoder.parameters()) + list(w_encoder.parameters()), lr=lr)

# Prefix sequences
def one_epoch(series):
    z_encoder.train()
    p_encoder.train()
    w_encoder.train()
    opt.zero_grad()
    list_zip = th.LongTensor(np.arange(len(date_dict))).to(device)
    list_poi = th.LongTensor(np.arange(len(poi_dict))).to(device)
    losses = 0.
    for i in range(10, series.shape[1]):
        batch_series = th.FloatTensor(series[:, :(i-ahead+1)]).to(device)
        batch_labels = th.FloatTensor(series[:, i]).to(device)
        wt_matrix = th.FloatTensor(visit_matrix[(i-ahead+1)].T/10.0).to(device)
        ez, em_z = z_encoder.forward(list_zip, batch_series)
        ep, em_p = p_encoder.forward(list_poi)
        wt = w_encoder.forward(em_z, em_p, wt_matrix)
        preds = ez + wt @ ep
        loss = th.mean((preds - batch_labels) ** 2)
        losses += loss
    tot_loss = losses
    tot_loss.backward()
    opt.step()
    print(f"Total loss {tot_loss.detach().cpu().item()}")
    #print("ez",ez.ravel().detach().cpu().numpy())
    #print("ep", ep.ravel().detach().cpu().numpy())
    #print("wt",wt.detach().cpu().numpy())

for ep in range(epochs):
    print(f"Epoch {ep+1}")
    one_epoch(train_series)

z_encoder.eval()
p_encoder.eval()
w_encoder.eval()
list_zip = th.LongTensor(np.arange(len(date_dict))).to(device)
list_poi = th.LongTensor(np.arange(len(poi_dict))).to(device)
losses = 0.
preds_all, gt = [], []
baselines, poi, wts = [], [], []
for i in range(10, train_series.shape[1]):
    batch_series = th.FloatTensor(train_series[:, :(i-ahead+1)]).to(device)
    batch_labels = th.FloatTensor(train_series[:, i]).to(device)
    wt_matrix = th.FloatTensor(visit_matrix[(i-ahead+1)].T/10.0).to(device)
    ez, em_z = z_encoder.forward(list_zip, batch_series)
    ep, em_p = p_encoder.forward(list_poi)
    wt = w_encoder.forward(em_z, em_p, wt_matrix)
    preds = ez + wt @ ep
    preds_all.append(preds.detach().cpu().numpy().squeeze())
    gt.append(batch_labels.detach().cpu().numpy().squeeze())
    baselines.append(ez.detach().cpu().numpy().squeeze())
    poi.append(ep.detach().cpu().numpy().squeeze())
    wts.append(wt.detach().cpu().numpy().squeeze())
preds_all= np.array(preds_all)
gt = np.array(gt)
baselines = np.array(baselines)
poi = np.array(poi)
wts = np.array(wts)

os.makedirs("./pred_dir", exist_ok=True)
with open(f"./pred_dir/preds_{ahead}.pkl", "wb") as f:
    pickle.dump({
        "preds": preds_all,
        "gt": gt,
        "baselines": baselines,
        "poi": poi,
        "wts": wts
    }, f)
