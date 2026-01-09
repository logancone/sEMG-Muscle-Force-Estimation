# Heavily inspired by code provided by Wang et al.
# https://doi.org/10.32604/iasc.2023.036701

import torch.nn as nn

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #Allows reference of train

from train import train_loop

# Construct the class
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # First conv layer
        self.conv_1 = nn.Sequential(
            nn.Conv1d(1, 16, 4, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        # Second conv layer
        self.conv_2 = nn.Sequential(
            nn.Conv1d(16, 32, 5, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2,2),
            nn.Dropout(p=0.07),
        )

        # Feedforward layer
        self.fcs = nn.Sequential(
            nn.Linear(6, 24),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(24,12),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(12,6),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(6, 1)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.fcs(x)

        return x
    
def train_function():
        model = CNN()
        train_loop(model, 50)
        # Call train func\

if __name__ == "__main__":
        train_function()