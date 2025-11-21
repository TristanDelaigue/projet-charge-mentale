import torch
import torch.nn as nn
import torch.optim as optim


class FCN(nn.Module):
    
    def __init__(self, input_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.conv1 = nn.Conv1d(input_size, 64)
        self.conv2 = nn.Conv1d(64, 64)
        self.gap = nn.AvgPool1d()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
    def forward(self, x_ecg, x_gsr, x_inf_ppg, x_pix_ppg):
        
        # ecg layer
        x_ecg = self.relu(self.conv1(x_ecg))
        x_ecg = self.relu(self.conv2(x_ecg))
        x_ecg = self.relu(self.conv2(x_ecg))
        x_ecg = self.gap(x_ecg)
        
        # gsr layer
        x_gsr = self.relu(self.conv1(x_gsr))
        x_gsr = self.relu(self.conv2(x_gsr))
        x_gsr = self.relu(self.conv2(x_gsr))
        x_gsr = self.gap(x_gsr)
        
        # inf ppg layer
        x_inf_ppg = self.relu(self.conv1(x_inf_ppg))
        x_inf_ppg = self.relu(self.conv2(x_inf_ppg))
        x_inf_ppg = self.relu(self.conv2(x_inf_ppg))
        x_inf_ppg = self.gap(x_inf_ppg)
        
        # pix ppg layer
        x_pix_ppg = self.relu(self.conv1(x_pix_ppg))
        x_pix_ppg = self.relu(self.conv2(x_pix_ppg))
        x_pix_ppg = self.relu(self.conv2(x_pix_ppg))
        x_pix_ppg = self.gap(x_pix_ppg)
        
        # concatenate all layers
        x = torch.cat(x_ecg, x_gsr, x_inf_ppg, x_pix_ppg, dim=1)
        x = self.softmax(self.relu(x))
        
        return x
        
        
    

