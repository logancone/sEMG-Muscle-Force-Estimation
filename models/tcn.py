import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

# from train import train_loop


# 1 Dimensional Chomping Layer to trim conv output
class Chomp1d(nn.Module):
    # chomp_size = padding size
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv_1 = nn.Sequential(

            weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, 
                                           stride=stride, padding=padding, dilation=dilation)),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.conv_2 = nn.Sequential(
            weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, 
                                           stride=stride, padding=padding, dilation=dilation)),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
        #                          self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.relu = nn.ReLU()
        # self.init_weights()

    # def init_weights(self):
    #     conv = self.conv_1[0]
    #     nn.init.normal_(conv.weight_v, 0, 0.01)
    #     self.conv2.weight.data.normal_(0, 0.01)
    #     if self.downsample is not None:
    #         self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.relu(x)
        # res = x if self.downsample is None else self.downsample(x)
        return x
    

class TemporalConvNet(nn.Module):
    # difference in two references
    def __init__(self, num_inputs, num_channels=None, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        if num_channels is None:
            num_channels = [8, 16, 32]

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
            
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1[:, :, -1])


# def train_function():
    
#     model = TCN(1, 1, [30]*8, 7, 0.0)
#     train_loop(model, 5)


# if __name__ == "__main__":
#         train_function()

