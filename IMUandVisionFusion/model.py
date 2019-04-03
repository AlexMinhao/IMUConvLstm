import torch
import torch.nn as nn
from torch.autograd import Variable
from definitions import *
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
            nn.Conv2d(in_channels=1, out_channels=NUM_FILTERS, kernel_size=(12, FILTER_SIZE), stride=(1, 3)),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=(FILTER_SIZE, FILTER_SIZE)),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.Dropout2d(0.5),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=(FILTER_SIZE, FILTER_SIZE)),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.Dropout2d(0.5),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=(1, FILTER_SIZE)),
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
            out = out.view(-1, 9*31, NUM_FILTERS) #CHANNELS_NUM_50


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