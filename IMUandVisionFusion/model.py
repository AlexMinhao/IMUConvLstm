import torch
import torch.nn as nn
from torch.autograd import Variable
from definitions import *
from GraphConv import ConvTemporalGraphical
import torch.nn.functional as F
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
    def __init__(self):
        super(ConvLSTM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=NUM_FILTERS, kernel_size=(FILTER_SIZE, 1)),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=(FILTER_SIZE, 1)),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.Dropout2d(0.5),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=(FILTER_SIZE, 1)),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.Dropout2d(0.5),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=(FILTER_SIZE, 1)),
            # nn.BatchNorm2d(NUM_FILTERS),
            # nn.Dropout2d(0.5),
            nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=(1, FILTER_SIZE)),
            # nn.BatchNorm2d(NUM_FILTERS),
            # nn.Dropout2d(0.5),
            nn.ReLU())
        self.lstm = nn.LSTM(NUM_FILTERS, NUM_UNITS_LSTM, NUM_LSTM_LAYERS, batch_first=True)
        if NO_NLSTM:
            self.fc = nn.Linear(64 * 105, NUM_CLASSES)
        else:
            self.fc = nn.Linear(NUM_UNITS_LSTM, NUM_CLASSES)

    def forward(self, x):
        # print (x.shape)
        out = self.conv1(x)
        # print (out.shape)
        out = self.conv2(out)
        # print (out.shape)
        out = self.conv3(out)
        # print (out.shape)
        out = self.conv4(out)
        # print (out.shape)
        # out = out.view(-1, NB_SENSOR_CHANNELS, NUM_FILTERS)
        if NO_NLSTM:
            out = out.view(-1, 64 * CHANNELS_NUM_50_TO_42)
        else:
            out = out.view(-1, 8*113, NUM_FILTERS) #CHANNELS_NUM_50


        h0 = Variable(torch.zeros(NUM_LSTM_LAYERS, out.size(0), NUM_UNITS_LSTM))
        c0 = Variable(torch.zeros(NUM_LSTM_LAYERS, out.size(0), NUM_UNITS_LSTM))
        if torch.cuda.is_available():
            h0, c0 = h0.cuda(), c0.cuda()

        # forward propagate rnn


        if NO_NLSTM:
            out = self.fc(out)
        else:
            out, _ = self.lstm(out, (h0, c0))
            #  out[:, -1, :] -> [100,11,128] ->[100,128]
            out = self.fc(out[:, -1, :])
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size = 1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding = 2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding = 1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding = 1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self,x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x,kernel_size=3, stride = 1, padding = 1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs,1)

class gcn_lstm(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 ):
        super(gcn_lstm, self).__init__()
        padding = ((kernel_size[0] - 1) // 2, 0)

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

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A

if __name__ == '__main__':
    model = ConvLSTM()
    print(model)  # net architecture

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