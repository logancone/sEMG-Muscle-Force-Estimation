# Inspired from https://github.com/ozancanozdemir/CNN-LSTM/blob/main/cnn-lstm.py

import torch
import torch.nn as nn

from train import train_loop

class CLSTM(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(CLSTM, self).__init__()

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=2, stride=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(64, 32, kernel_size=1, stride=1, padding=1)
        self.batch1 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(32, 32, kernel_size=1, stride=1, padding=1)
        self.batch2 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()

        self.lstm = nn.LSTM(input_size=23, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(32*hidden_size, output_size)


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch1(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.batch2(x)
        x = self.relu3(x)
        x, h = self.lstm(x)
        x = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]))
        x = self.fc1(x)

        return x
        

def train_function():
        model = CLSTM(1, 1, 2, 5)
        train_loop(model, 50)
        # Call train func\

if __name__ == "__main__":
        train_function()