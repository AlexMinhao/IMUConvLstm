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
    def __init__(self, in_channels, num_class):
        super(ConvLSTM, self).__init__()

        self.inception1 = InceptionA(in_channels, 16)
        self.inception2 = InceptionA(64, 16)
        self.fcn = nn.Conv2d(80, NUM_FILTERS, kernel_size=1)
        self.lstm = nn.LSTM(NUM_FILTERS, NUM_UNITS_LSTM, NUM_LSTM_LAYERS, dropout=0.05, batch_first=True)
        self.out = nn.Linear(32*24*113, num_class)


    def forward(self, x):
        out = self.inception1(x)
        out = self.inception2(out)
        out = self.inception2(out)
        out = self.fcn(out)

        # out = out.permute(0, 3, 2, 1).contiguous()
        # N,  sentence, word, channel = out.size()    # 100 113 24 64
        # out = out.view(N*sentence, word, channel)
        #
        # h0 = Variable(torch.zeros(NUM_LSTM_LAYERS, out.size(0), NUM_UNITS_LSTM))
        # c0 = Variable(torch.zeros(NUM_LSTM_LAYERS, out.size(0), NUM_UNITS_LSTM))
        # if torch.cuda.is_available():
        #     h0, c0 = h0.cuda(), c0.cuda()
        #
        # out, _ = self.lstm(out, (h0, c0))
        #     #  24 11300 128
        # out = out[:, -1, :]
        # out = out.view(N, sentence, NUM_UNITS_LSTM) #100 113 24 128

        out = out.view(out.size(0), -1)

        out = self.out(out)
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
        self.branch1x1 = BasicConv2d(in_channels, 16, kernel_size = 1)  # 64 x 24 x 113

        self.branch5x5_1 = BasicConv2d(in_channels, 32, kernel_size=1)  # 48 x 24 x 113
        self.branch5x5_2 = BasicConv2d(32, 16, kernel_size=(5, 1), padding=(2, 0))

        self.branch7x7_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(32, 16, kernel_size=(7, 1), padding=(3, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(32, 64, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_3 = BasicConv2d(64, 16, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        # branch1x1 = self.branch1x1(x)

        branch5x1 = self.branch5x5_1(x)
        branch5x1 = self.branch5x5_2(branch5x1)

        branch7x1 = self.branch7x7_1(x)
        branch7x1 = self.branch7x7_2(branch7x1)

        branch3x1dbl = self.branch3x3dbl_1(x)
        branch3x1dbl = self.branch3x3dbl_2(branch3x1dbl)
        branch3x1dbl = self.branch3x3dbl_3(branch3x1dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=(5, 1), stride=1, padding=(2, 0))
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch5x1, branch3x1dbl, branch7x1, branch_pool]
        return torch.cat(outputs, 1)



if __name__ == '__main__':
    in_channels = 1
    num_class = 18
    edge_importance_weighting = True
    model = ConvLSTM(in_channels, num_class)
    print(model)  # net architecture
    # N C T V
    x = torch.zeros(100, 1, 24, 113)
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