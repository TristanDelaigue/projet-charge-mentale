from scipy.signal import resample
import pandas as pd
from fcn import FCN
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split

x_ecg = []
x_gsr = []
x_inf_ppg = []
x_pix_ppg = []
y = []

for folder_name in os.walk("data/MAUS/Data/Raw_data/"):
    if folder_name[0][-1] != '/':
        for trial in pd.read_csv(f"{folder_name[0]}/inf_ecg.csv").to_numpy().transpose():
            x_ecg.append(list(trial))
        for trial in pd.read_csv(f"{folder_name[0]}/inf_gsr.csv").to_numpy().transpose():
            x_gsr.append(list(trial))
        for trial in pd.read_csv(f"{folder_name[0]}/inf_ppg.csv").to_numpy().transpose():
            x_inf_ppg.append(list(trial))
        for trial in  pd.read_csv(f"{folder_name[0]}/pixart.csv").to_numpy().transpose():
            x_pix_ppg.append(list(trial))
        for trial in pd.read_csv(f"data/MAUS/Subjective_rating/{folder_name[0][-3:]}/NASA_TLX.csv").iloc[7, 1:7].to_numpy():
            y.append(trial)

# resample data to 4Hz on 30 seconds
resample_size = 120
x_ecg_res = [resample(x, resample_size) for x in x_ecg]
x_gsr_res = [resample(x, resample_size) for x in x_gsr]
x_inf_ppg_res = [resample(x, resample_size) for x in x_inf_ppg]
#x_pix_ppg_res = resample(x_pix_ppg, resample_size) TODO make this work (not all things of the same size)

x_ecg_res_train, x_ecg_res_test, x_gsr_res_train, x_gsr_res_test, x_inf_ppg_res_train, x_inf_ppg_res_test, y_train, y_test = train_test_split(
    x_ecg_res,
    x_gsr_res,
    x_inf_ppg_res,
    y,
    train_size=0.8
)

x_ecg_res_train_loader = torch.utils.data.DataLoader(x_ecg_res_train, shuffle=True, batch_size=12)
x_ecg_res_test_loader = torch.utils.data.DataLoader(x_ecg_res_test, shuffle=True, batch_size=12)
x_gsr_res_train_loader = torch.utils.data.DataLoader(x_gsr_res_train, shuffle=True, batch_size=12)
x_gsr_res_test_loader = torch.utils.data.DataLoader(x_gsr_res_test, shuffle=True, batch_size=12)
x_inf_res_ppg_train_loader = torch.utils.data.DataLoader(x_inf_ppg_res_train, shuffle=True, batch_size=12)
x_inf_res_ppg_test_loader = torch.utils.data.DataLoader(x_inf_ppg_res_test, shuffle=True, batch_size=12)
y_train_loader = torch.utils.data.DataLoader(y_train, shuffle=True, batch_size=12)
y_test_loader = torch.utils.data.DataLoader(y_test, shuffle=True, batch_size=12)


