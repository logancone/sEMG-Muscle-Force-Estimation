# Inspired from https://github.com/ozancanozdemir/CNN-LSTM/blob/main/cnn-lstm.py

import torch
import torch.nn as nn

# from train import train_loop

class CLSTM(nn.Module):
    
    def __init__(self, hidden_size, num_layers):
        super(CLSTM, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=2, stride=1),
            nn.ReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=1, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.conv_3 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=1, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        self.fcs = nn.Linear(hidden_size, 1)


    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)

        x = x.permute(0, 2, 1)

        
        x, h = self.lstm(x)
        x = x[:, -1, :]  
        x = self.fcs(x)

        return x
        

# def train_function():
#         model = CLSTM(1, 1, 2, 5)
#         train_loop(model, 5)
#         # Call train func\

# if __name__ == "__main__":
#         train_function()