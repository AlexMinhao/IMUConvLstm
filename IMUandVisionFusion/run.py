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
from process_data_new import *
from import_dataset import *


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
    data_y_temp = sliding_window(data_y, ws, ss) # (46495,24,113)
    data_y = np.asarray([[i[-1]] for i in data_y_temp]) # (46495,1)
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)

def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x



if __name__ == '__main__':
    path = 0
    if SEVER == 1:
        path = os.path.join(os.path.dirname(os.getcwd()), r'OPPORTUNITY\OppSegBySubjectGestures.data')
    else:
        path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), r'OPPORTUNITY\OppSegBySubjectGestures.data')

    # preprocess_data(dr,dp)

    print("Loading data...")

    if DATAFORMAT:

        Opp = OPPORTUNITY(path)
        X_train, y_train, X_test, y_test = Opp.load() #load_dataset(dp)
        assert NB_SENSOR_CHANNELS == X_train.shape[2]

        print(" ..after sliding window (training): inputs {0}, targets {1}".format(X_train.shape, y_train.shape))
        print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))
        #
        # Data is reshaped since the input of the network is a 4 dimension tensor
        X_test = X_test.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
        X_train = X_train.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))  #inputs (46495, 1, 24, 113), targets (46495,)
        X_train.astype(np.float32), y_train.reshape(len(y_train)).astype(np.uint8)
        # X_train = X_train[1:300]
        # y_train = y_train[1:300]
        # X_test = X_test[1:300]
        # y_test = y_test[1:300]
        print(" after reshape: inputs {0}, targets {1}".format(X_train.shape, y_train.shape))
    else:
        trainset = OPPORTUNITY('D:/Research/DeepConvLSTM/OPPORTUNITY/gestures.data', train=True)
        testset = OPPORTUNITY('D:/Research/DeepConvLSTM/OPPORTUNITY/gestures.data', train=False)

    model = ConvLSTM()
    if torch.cuda.is_available():
        model.cuda()
        print("Model on gpu")

    # If use CrossEntropyLoss，softmax wont be used in the final layer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=10e-5)

    if DATAFORMAT:
        if CONTAIN_NULLCLASS:
            X_train = list(X_train)
            y_train = list(y_train)
            training_set = []
            for i in range(len(y_train)):
                x = X_train[i]
                y = y_train[i]
                xy = (x, y)
                training_set.append(xy)

            X_test = list(X_test)
            y_test = list(y_test)
            testing_set = []
            for j in range(len(y_test)):
                x = X_test[j]
                y = y_test[j]
                xy = (x, y)
                testing_set.append(xy)

        else:
            nullclass_index = np.argwhere(y_train == 0)
            X_train = list(np.delete(X_train, nullclass_index, axis=0))
            y_train = list(np.delete(y_train, nullclass_index))

            dataset = []
            for i in range(len(y_train)):
                x = X_train[i]
                y = y_train[i]
                xy = (x, y)
                dataset.append(xy)

            X_test = list(X_test)
            y_test = list(y_test)
            testing_set = []
            for j in range(len(y_test)):
                x = X_test[j]
                y = y_test[j]
                xy = (x, y)
                testing_set.append(xy)


    train_loader = DataLoader(dataset=training_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=testing_set, batch_size=BATCH_SIZE, shuffle=True)



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

            seqs = get_variable(seqs.float())
            labels = get_variable(labels.long())

            outputs = model(seqs)
            # print(outputs[1:3, :])
            labels = labels.squeeze()
            loss = loss_function(outputs, labels)
            losses.update(loss.item() / BATCH_SIZE, 1)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i == 2:
                ttt = 1

            _, preds = torch.max(outputs.data, 1)
            # print(preds[1:10])

            total += labels.size(0)
            labels_reshape = labels
            correct += (preds == labels_reshape.data).sum().item()
            f1_train.update(f1_score(labels_reshape, preds, average='micro'))
            # measure elapsed time
            batch_time.update(time() - end)

            end = time()
            if (i + 1) % 1 == 0:
                print(
                    'Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Acc: %.4f, Time: %.3f, F1-score: %.3f, F1-score.avg: %.3f '
                    % (epoch + 1, EPOCH, i + 1, len(training_set) // BATCH_SIZE, loss.item(), 100 * correct / total,
                       batch_time.val, f1_train.val, f1_train.avg))

    print('Accuracy of the model  {0} %, F1-score: {1}'.format(100 * correct / total, f1_train.avg))
    # Test the model

    model.eval()
    correct = 0
    total = 0
    for i, (seqs, labels) in enumerate(test_loader):
        # measure data loading time
        data_time.update(time() - end)

        seqs = get_variable(seqs.float())
        labels = get_variable(labels.long())

        outputs = model(seqs)
        labels = labels.squeeze()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        labels_reshape = labels
        correct += (predicted == labels_reshape.data).sum()
        f1_test.update(f1_score(labels_reshape, predicted, average='micro'))
    print('Test Accuracy of the model  {0}%, F1-score {1}%'.format(100 * correct / total, f1_test.avg))
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for i, (seqs, labels) in enumerate(train_loader):
    #         seqs = get_variable(seqs)
    #         labels = get_variable(labels.long().cuda())
    #
    #         outputs = model(seqs)
    #         _, predicted = torch.max(outputs, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #         f1_test.update(f1_score(labels, predicted, average='micro'))


    # Save the model checkpoint
