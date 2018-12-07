import os
import numpy as np
import pickle as cp
from sliding_window import sliding_window
import torch
from model import ConvLSTM
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as data
from definitions import *
from helper import *
from sklearn.metrics import f1_score



def preprocess_data(dataset_raw, dataset_processed):

    path = 'python preprocess_data.py -i' + ' ' +dataset_raw + ' '+ '-o' + ' ' + dataset_processed
    print(path)
    os.system(path)


def load_dataset(filename):

    f = open(filename, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    #np.savetxt('D:/Research/DeepConvLSTM/OPPORTUNITY/new.csv', y_train, delimiter=',')

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))
    # ..reading instances: train (557963, 113), test (118750, 113)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test

def opp_sliding_window(data_x, data_y, ws, ss):
    a = data_x.shape[1]   #data_x.shape[0] = 557963  data_x.shape[1] = 113
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y_temp = sliding_window(data_y, ws, ss)
    data_y = np.asarray([[i[-1]] for i in data_y_temp])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)

def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x

if __name__ == '__main__':
    if SEVER == 1:
        dr = '/home/fyhuang/ConvLSTM/OPPORTUNITY/OpportunityUCIDataset.zip'
        dp = '/home/fyhuang/ConvLSTM/OPPORTUNITY/oppChallenge_gestures.data'
    else:
        dr = 'D:/Research/DeepConvLSTM/OPPORTUNITY/OpportunityUCIDataset.zip'
        dp = 'D:/Research/DeepConvLSTM/OPPORTUNITY/oppChallenge_gestures.data'

    #preprocess_data(dr,dp)

    print("Loading data...")
    X_train, y_train, X_test, y_test = load_dataset(dp)
    assert NB_SENSOR_CHANNELS == X_train.shape[1]
    # print(" Training: inputs {0}, targets {1}".format(X_train.shape, y_train.shape))   Training: inputs (557963, 113), targets (557963,)
    # print(" Testing: inputs {0}, targets {1}".format(X_test.shape, y_test.shape))       Testing: inputs (118750, 113), targets (118750,)

    # Sensor data is segmented using a sliding window mechanism
    X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    print(" ..after sliding window (training): inputs {0}, targets {1}".format(X_train.shape, y_train.shape))
    print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))
    #
    # Data is reshaped since the input of the network is a 4 dimension tensor
    X_test = X_test.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
    X_train = X_train.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))  #inputs (46495, 1, 24, 113), targets (46495,)
    print(" after reshape: inputs {0}, targets {1}".format(X_train.shape, y_train.shape))

    model = ConvLSTM()
    if torch.cuda.is_available():
        model.cuda()
        print("Model on gpu")

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=10e-4, weight_decay=0.9)

    X_train = list(X_train)
    y_train = list(y_train)
    dataset = []
    for i in range(len(y_train)):
        x = X_train[i]
        y = y_train[i]
        xy = (x, y)
        dataset.append(xy)

    X_test = list(X_test)
    y_test = list(y_test)
    testset = []
    for j in range(len(y_test)):
        x = X_test[j]
        y = y_test[j]
        xy = (x, y)
        testset.append(xy)


    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_loader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    f1_train = AverageMeter()
    f1_test = AverageMeter()

    end = time()
    correct = 0
    total = 0

# training and testing
    for epoch in range(EPOCH):
        for i, (seqs, labels) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time() - end)

            seqs = get_variable(seqs)
            labels = get_variable(labels.long())

            outputs = model(seqs)
            loss = loss_function(outputs, labels)
            losses.update(loss.item() / BATCH_SIZE, 1)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            f1_train.update(f1_score(labels, preds, average='micro'))
            # measure elapsed time
            batch_time.update(time() - end)
            end = time()
            if (i + 1) % 1 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Acc: %.4f, Time: %.3f, F1-score: %.3f, F1-score.avg: %.3f '
                      % (epoch + 1, EPOCH, i + 1, len(dataset) // BATCH_SIZE, loss.item(), 100 * correct / total, batch_time.val, f1_train.val, f1_train.avg))

    print('Train Accuracy of the model: {0} %, F1-score: {1}'.format(100 * correct / total, f1_train.avg))
    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (seqs, labels) in enumerate(train_loader):

            seqs = get_variable(seqs)
            labels = get_variable(labels.long().cuda())

            outputs = model(seqs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            f1_test.update(f1_score(labels, predicted, average='micro'))

        print('Test Accuracy of the model: {0} %, F1-score: {1}'.format(100 * correct / total, f1_test.avg))

    # Save the model checkpoint
