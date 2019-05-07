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
        self.Conv2d_1a_5x1 = BasicConv2d(in_channels, NUM_FILTERS, kernel_size=(5,1), padding=(2, 0))
        self.Conv2d_2a_5x1 = BasicConv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=(5,1), padding=(2, 0))
        self.inceptionA = InceptionA(NUM_FILTERS, pool_features=32)
        self.inceptionC = InceptionC(160, channels_7x1=64)
        self.inceptionB = InceptionB(64)
        self.Conv2d_3a_5x1 = BasicConv2d(NUM_FILTERS*2, NUM_FILTERS, kernel_size=1)

        self.fcn = nn.Conv2d(256, NUM_FILTERS, kernel_size=1)
        self.lstm = nn.LSTM(NUM_FILTERS, NUM_UNITS_LSTM, NUM_LSTM_LAYERS, dropout=0.05, batch_first=True)
        self.out = nn.Linear(64*113, num_class)


    def forward(self, x):
        out = self.Conv2d_1a_5x1(x)
        # 100 x 64 x 24 x 113
        out = self.Conv2d_2a_5x1(out)
        # 100 x 64 x 24 x 113
        out = self.inceptionA(out)
        # 100 x 160 x 24 x 113
        out = self.inceptionC(out)
        # 100 x 256 x 24 x 113
        out = self.fcn(out)
        # 100 x 64 x 24 x 113
        out = F.avg_pool2d(out, kernel_size=(3,1),stride = (3,1))
        # 100 x 64 x 8 x 113
        out = self.inceptionB(out)

        out = F.avg_pool2d(out, kernel_size=(4, 1))

        out = self.Conv2d_3a_5x1(out)
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
        self.branch1x1 = BasicConv2d(in_channels, 32, kernel_size = 1)  # 64 x 24 x 113

        self.branch5x5_1 = BasicConv2d(in_channels, 32, kernel_size=1)  # 48 x 24 x 113
        self.branch5x5_2 = BasicConv2d(32, 32, kernel_size=(5, 1), padding=(2, 0))

        self.branch7x7_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(32, 32, kernel_size=(7, 1), padding=(3, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(32, 64, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_3 = BasicConv2d(64, 32, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x1 = self.branch5x5_1(x)
        branch5x1 = self.branch5x5_2(branch5x1)

        branch7x1 = self.branch7x7_1(x)
        branch7x1 = self.branch7x7_2(branch7x1)

        branch3x1dbl = self.branch3x3dbl_1(x)
        branch3x1dbl = self.branch3x3dbl_2(branch3x1dbl)
        branch3x1dbl = self.branch3x3dbl_3(branch3x1dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=(5, 1), stride=1, padding=(2, 0))
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x1, branch3x1dbl, branch7x1, branch_pool]
        return torch.cat(outputs, 1)

class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x1 = BasicConv2d(in_channels, 32, kernel_size=(3, 1), padding=(1,0), stride = (2,1))

        self.branch3x1dbl_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.branch3x1dbl_2 = BasicConv2d(32, 32, kernel_size=(3, 1), padding=(1,0))
        self.branch3x1dbl_3 = BasicConv2d(32, 32, kernel_size=(3, 1), padding=(1,0), stride=(2,1))

    def forward(self, x):
        branch3x1 = self.branch3x1(x)

        branch3x1dbl = self.branch3x1dbl_1(x)
        branch3x1dbl = self.branch3x1dbl_2(branch3x1dbl)
        branch3x1dbl = self.branch3x1dbl_3(branch3x1dbl)

        branch_pool = F.max_pool2d(x, kernel_size=(3, 1), padding=(1,0), stride=(2,1))

        outputs = [branch3x1, branch3x1dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x1):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        c7 = channels_7x1
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7_3 = BasicConv2d(c7, 64, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 64, kernel_size=(7, 1), padding=(3, 0))

        self.branch_pool = BasicConv2d(in_channels, 64, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=(7,1), stride=1, padding=(3,0))
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x



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