import torch
import torch.nn as nn
from torch.autograd import Variable
from definitions import *
from GraphConv import ConvTemporalGraphical
from graph import Graph
import torch.nn.functional as F
from utils import get_variable
import numpy as np
# ### Define the Lasagne network
'''
Sensor data are processed by four convolutional layer which allow to learn features from the data.
Two dense layers then perform a non-lineartransformation which yields the classification outcome with a softmax logistic regresion output layer
'''



'''
    BATCH_SIZE, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS
input[  100     1        12                  113   ]          =>  
       NUM_FILTERS     (FILTER_SIZE, 1)
Layer1[     64               5       1 ]                      => 
       NUM_FILTERS     (FILTER_SIZE, 1)
Layer2[     64               5       1 ]                      => 
       NUM_FILTERS     (FILTER_SIZE, 1)
Layer3[     64               5       1 ]                      => 
       NUM_FILTERS     (FILTER_SIZE, 1)
Layer4[     64               5       1 ]                      => 
       NUM_UNITS_LSTM
Layer5[     128       ]                                       => 
       NUM_UNITS_LSTM
Layer6[     128       ]                                       => 
'''

class ConvLSTM(nn.Module):
    def __init__(self, in_channels, num_class):
        super(ConvLSTM, self).__init__()
        self.graph = Graph()
        self.A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.A = get_variable(self.A)
        # build networks
        spatial_kernel_size = self.A.size(0)
        temporal_kernel_size = 5
        kernel_size = (temporal_kernel_size, spatial_kernel_size)  # 5 * 2
        self.data_bn = nn.BatchNorm1d(in_channels*self.A.size(1))
        self.gcn_lstm_networks = nn.ModuleList((
            gcn_lstm(in_channels, NUM_FILTERS, kernel_size, 1),
            gcn_lstm(NUM_FILTERS, NUM_FILTERS, kernel_size, 1),
            gcn_lstm(NUM_FILTERS, 2 * NUM_FILTERS, kernel_size, 1),
            gcn_lstm(2*NUM_FILTERS, 2 * NUM_FILTERS, kernel_size, 1),
            gcn_lstm(2*NUM_FILTERS, 4*NUM_FILTERS, kernel_size, 1),
            gcn_lstm(4*NUM_FILTERS, 2*NUM_FILTERS, kernel_size, 1),
            gcn_lstm(2*NUM_FILTERS, NUM_FILTERS, kernel_size, 1),
        ))

        self.edge_importance = [1] * len(self.gcn_lstm_networks)
        self.fcn = nn.Conv2d(NUM_FILTERS, num_class, kernel_size=1)



    def forward(self, x):
        # data normalization
        # 100 9 24 7
        N, C, T, V = x.size()
        xa = x[:, 0:3, :, :] # 100 3 24 7
        xg = x[:, 3:6, :, :]
        xm = x[:, 6:9, :, :]
        xa = xa.view(N, V * 3, T)
        xg = xg.view(N, V * 3, T)
        xm = xm.view(N, V * 3, T)
        xa = self.data_bn(xa)
        xg = self.data_bn(xg)
        xm = self.data_bn(xm)
        xa = xa.view(N, V, 3, T)
        xa = xa.permute(0, 2, 3, 1).contiguous()
        xg = xg.view(N, V, 3, T)
        xg = xg.permute(0, 2, 3, 1).contiguous()
        xm = xm.view(N, V, 3, T)
        xm = xm.permute(0, 2, 3, 1).contiguous()


        for gcn, importance in zip(self.gcn_lstm_networks, self.edge_importance):
            xa, _ = gcn(xa, self.A * importance)
            xg, _ = gcn(xg, self.A * importance)
            xm, _ = gcn(xm, self.A * importance)

        xa = F.avg_pool2d(xa, xa.size()[2:])
        xa = xa.view(N,  -1, 1, 1)
        xa = self.fcn(xa)
        xa = xa.view(xa.size(0), -1)

        xg = F.avg_pool2d(xg, xg.size()[2:])
        xg = xg.view(N, -1, 1, 1)
        xg = self.fcn(xg)
        xg = xg.view(xg.size(0), -1)

        xm = F.avg_pool2d(xm, xm.size()[2:])
        xm = xm.view(N, -1, 1, 1)
        xm = self.fcn(xm)
        xm = xm.view(xm.size(0), -1)

        out = xa + xg + xm

        return out


class gcn_lstm(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 ):
        super(gcn_lstm, self).__init__()
        padding = ((kernel_size[0]-1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        #res = self.residual(x)
        x, A = self.gcn(x, A) #100 64 24 7
        x = self.tcn(x)

        return self.relu(x), A





if __name__ == '__main__':
    in_channels = 3
    num_class = 18
    edge_importance_weighting = True
    model = ConvLSTM(in_channels,num_class)
    print(model)  # net architecture
    # N C T V
    x = torch.zeros(100, 9, 24, 7)
    x = Variable(x)
    out = model(x)
    print(out[1])
    '''
    ConvLSTM(
  (conv1): Sequential(
    (0): Conv2d(1, 64, kernel_size=(5, 1), stride=(1, 1))
    (1): ReLU()
  )
  (conv2): Sequential(
    (0): Conv2d(64, 64, kernel_size=(5, 1), stride=(1, 1))
    (1): ReLU()
  )
  (conv3): Sequential(
    (0): Conv2d(64, 64, kernel_size=(5, 1), stride=(1, 1))
    (1): ReLU()
  )
  (conv4): Sequential(
    (0): Conv2d(64, 64, kernel_size=(5, 1), stride=(1, 1))
    (1): ReLU()
  )
  (lstm): LSTM(64, 128, num_layers=2, batch_first=True)
  (fc): Linear(in_features=128, out_features=18, bias=True)
)
    '''