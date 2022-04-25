#Import
import torch
from pathlib import Path
from datetime import timedelta

import dateutil
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from TaPR_pkg import etapr
from scipy import signal

#Parameter
N_HIDDENS = 100
N_LAYERS = 3
BATCH_SIZE = 512
TIMESTAMP_FIELD = "time"
IDSTAMP_FIELD = 'id'
ATTACK_FIELD = "attack"
STRIDE = 100
WINDOW_GIVEN = 39
WINDOW_SIZE = WINDOW_GIVEN + 1
THRESHOLD = 0.019200000000000002

valid_columns_in_train_dataset = ['P1_B2004', 'P1_B2016', 'P1_B3004', 'P1_B3005', 'P1_B4002', 'P1_B4005',
       'P1_B400B', 'P1_B4022', 'P1_FCV01D', 'P1_FCV01Z', 'P1_FCV02D',
       'P1_FCV02Z', 'P1_FCV03D', 'P1_FCV03Z', 'P1_FT01', 'P1_FT01Z', 'P1_FT02',
       'P1_FT02Z', 'P1_FT03', 'P1_FT03Z', 'P1_LCV01D', 'P1_LCV01Z', 'P1_LIT01',
       'P1_PCV01D', 'P1_PCV01Z', 'P1_PCV02D', 'P1_PCV02Z', 'P1_PIT01',
       'P1_PIT02', 'P1_PP01AD', 'P1_PP01AR', 'P1_PP01BD', 'P1_PP01BR',
       'P1_PP02D', 'P1_PP02R', 'P1_STSP', 'P1_TIT01', 'P1_TIT02', 'P2_24Vdc',
       'P2_ASD', 'P2_AutoGO', 'P2_CO_rpm', 'P2_Emerg', 'P2_HILout', 'P2_MSD',
       'P2_ManualGO', 'P2_OnOff', 'P2_RTR', 'P2_SIT01', 'P2_SIT02',
       'P2_TripEx', 'P2_VT01', 'P2_VTR01', 'P2_VTR02', 'P2_VTR03', 'P2_VTR04',
       'P2_VXT02', 'P2_VXT03', 'P2_VYT02', 'P2_VYT03', 'P3_FIT01', 'P3_LCP01D',
       'P3_LCV01D', 'P3_LH', 'P3_LIT01', 'P3_LL', 'P3_PIT01', 'P4_HT_FD',
       'P4_HT_LD', 'P4_HT_PO', 'P4_HT_PS', 'P4_LD', 'P4_ST_FD', 'P4_ST_GOV',
       'P4_ST_LD', 'P4_ST_PO', 'P4_ST_PS', 'P4_ST_PT01', 'P4_ST_TT01']

tag_min = pd.Series([0.02978, 0.94729, 369.75601, 890.07843, 31.413429999999998, 
                     0.0, 25.02598, 34.20232, 0.0, 0.26093, 0.0, -1.89819, 45.71051, 
                     46.174620000000004, -15.602129999999999, 0.0, 4.80653, 25.02598, 
                     187.14902, 865.69977, 2.1225099999999997, 0.28381, 355.98859, 
                     24.89728, 25.537110000000002, 12.0, 11.76605, 0.8691399999999999, 
                     0.17029, 540833.0, 540833.0, 0.0, 0.0, 1.0, 1.0, 1.0, 34.68933, 
                     34.97925, 28.01254, 0.0, 1.0, 53951.0, 0.0, 653.75977, 
                     763.1932400000001, 0.0, 1.0, 2880.0, 749.0, 748.93372, 1.0, 
                     11.76047, 10.0, 10.0, 10.0, 10.0, -4.4009, -2.1603, 2.4162,
                     4.5149, -27.0, -8.0, -288.0, 70.0, 5047.0, 20.0, -24.0, 
                     -0.04962, -0.0072299999999999994, 0.054229999999999993, 0.0, 
                     231.13788, -0.05244, 12625.0, 229.87201000000002, 
                     233.56122000000002, 0.0, 9914.0, 27530.0], index = valid_columns_in_train_dataset)

tag_max = pd.Series([0.10196000000000001, 2.0523, 447.83437999999995, 1121.9411599999999, 
                    33.6555, 100.0, 2857.54565, 38.650929999999995, 100.0, 100.0, 100.0, 
                    97.38311999999999, 74.18626, 75.37231, 473.06060999999994, 1381.9574, 
                    2006.56897, 2857.54565, 332.87042, 1147.55933, 28.336340000000003, 
                    28.535459999999997, 459.44928, 100.0, 100.0, 12.0, 12.04071, 2.39578, 
                    2.34695, 540833.0, 540833.0, 0.0, 0.0, 1.0, 1.0, 1.0, 36.94763, 40.4419, 
                    28.04642, 0.0, 1.0, 54215.0, 0.0, 775.1648, 763.1932400000001, 0.0, 
                    1.0, 2880.0, 828.0, 828.22137, 1.0, 12.06294, 10.0, 10.0, 10.0, 10.0, 
                    -1.7734, 0.2445, 5.156000000000001, 7.3006, 5443.0, 13816.0, 18416.0, 
                    70.0, 19730.0, 20.0, 3871.0, 0.035519999999999996, 83.10184, 86.71515, 
                    10.0, 499.76489000000004, 0.05035, 26927.0, 499.62018, 498.86065999999994, 
                    50.0, 10330.0, 27629.0], index = valid_columns_in_train_dataset)

#Class
class StackedGRU(torch.nn.Module):
    def __init__(self, n_tags):
        super().__init__()
        self.rnn = torch.nn.GRU(
            input_size=n_tags,
            hidden_size=N_HIDDENS,
            num_layers=N_LAYERS,
            bidirectional=True,
            dropout=0,
        )
        self.fc = torch.nn.Linear(N_HIDDENS * 2, n_tags)

    def forward(self, x):
        x = x.transpose(0, 1)  # (batch, seq, params) -> (seq, batch, params)
        self.rnn.flatten_parameters()
        outs, _ = self.rnn(x)
        out = self.fc(outs[-1])
        return x[0] + out
    
class HaiDataset(Dataset):
    def __init__(self, timestamps, df, stride=1, attacks=None):
        self.ts = np.array(timestamps)
        self.tag_values = np.array(df, dtype=np.float32)
        self.valid_idxs = []
        for L in trange(len(self.ts) - WINDOW_SIZE + 1):
            R = L + WINDOW_SIZE - 1
            if dateutil.parser.parse(self.ts[R]) - dateutil.parser.parse(
                self.ts[L]
            ) == timedelta(seconds=WINDOW_SIZE - 1):
                self.valid_idxs.append(L)
        self.valid_idxs = np.array(self.valid_idxs, dtype=np.int32)[::stride]
        self.n_idxs = len(self.valid_idxs)
        print(f"# of valid windows: {self.n_idxs}")
        if attacks is not None:
            self.attacks = np.array(attacks, dtype=np.float32)
            self.with_attack = True
        else:
            self.with_attack = False

    def __len__(self):
        return self.n_idxs

    def __getitem__(self, idx):
        i = self.valid_idxs[idx]
        last = i + WINDOW_SIZE - 1
        item = {"attack": self.attacks[last]} if self.with_attack else {}
        item["ts"] = self.ts[i + WINDOW_SIZE - 1]
        item["given"] = torch.from_numpy(self.tag_values[i : i + WINDOW_GIVEN])
        item["answer"] = torch.from_numpy(self.tag_values[last])
        return item
    
#function
def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())

def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

def normalize(df):
    ndf = df.copy()
    for c in df.columns:
        if tag_min[c] == tag_max[c]:
            ndf[c] = df[c] - tag_min[c]
        else:
            ndf[c] = (df[c] - tag_min[c]) / (tag_max[c] - tag_min[c])
    return ndf

def inference(dataset, model, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    ts, dist, att = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            given = batch["given"].cuda()
            answer = batch["answer"].cuda()
            guess = model(given)
            ts.append(np.array(batch["ts"]))
            dist.append(torch.abs(answer - guess).cpu().numpy())
            try:
                att.append(np.array(batch["attack"]))
            except:
                att.append(np.zeros(batch_size))
            
    return (
        np.concatenate(ts),
        np.concatenate(dist),
        np.concatenate(att),
    )

def put_labels(distance, threshold):
    xs = np.zeros_like(distance)
    xs[distance > threshold] = 1
    return xs

def fill_blank(check_ts, labels, total_ts):
    def ts_generator():
        for t in total_ts:
            yield dateutil.parser.parse(t)

    def label_generator():
        for t, label in zip(check_ts, labels):
            yield dateutil.parser.parse(t), label

    g_ts = ts_generator()
    g_label = label_generator()
    final_labels = []

    try:
        current = next(g_ts)
        ts_label, label = next(g_label)
        while True:
            if current > ts_label:
                ts_label, label = next(g_label)
                continue
            elif current < ts_label:
                final_labels.append(0)
                current = next(g_ts)
                continue
            final_labels.append(label)
            current = next(g_ts)
            ts_label, label = next(g_label)
    except StopIteration:
        return np.array(final_labels, dtype=np.int8)
    
#===================================================================================================
#===================================================================================================
#Start
#===================================================================================================
#===================================================================================================

#Get Dataset
#==============================================================================
test_dataset = sorted([x for x in Path("../data/test/").glob("*.csv")])
test_df_raw = dataframe_from_csvs(test_dataset).iloc[43201:]
test_df = normalize(test_df_raw[valid_columns_in_train_dataset]).ewm(alpha=0.9).mean()
hai_dataset_test = HaiDataset(
    test_df_raw[TIMESTAMP_FIELD], test_df, attacks=test_df_raw[ATTACK_FIELD]
)

# #Get Model
# #==============================================================================
model = StackedGRU(n_tags=test_df.shape[1])
model.cuda()

with open("../model/model.pt", "rb") as f:
    saved_model = torch.load(f)

model.load_state_dict(saved_model["state"])

# # #Predict
# # #==============================================================================
model.eval()
check_ts, check_dist, check_att = inference(hai_dataset_test, model, BATCH_SIZE)

anomaly_score = np.mean(check_dist, axis=1)
beta, alpha = signal.butter(1, 0.1, btype='lowpass')
anomaly_score_smooth = signal.filtfilt(beta, alpha, anomaly_score)

labels = put_labels(anomaly_score_smooth, THRESHOLD)
attack_labels = put_labels(np.array(test_df_raw[ATTACK_FIELD]), threshold=0.5)

final_labels = fill_blank(check_ts, labels, np.array(test_df_raw[TIMESTAMP_FIELD]))

TaPR = etapr.evaluate(anomalies=attack_labels, predictions=final_labels)
print(f"F1: {TaPR['f1']:.3f} (TaP: {TaPR['TaP']:.3f}, TaR: {TaPR['TaR']:.3f})")
print(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
print(f"Detected anomalies: {TaPR['Detected_Anomalies']}")