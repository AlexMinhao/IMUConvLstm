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
from utils import Logger
from utils import get_acc
from train import *
from validation import *
from test import *
from gan_augmentation import generator

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

# def checkpoint(epoch):
#     model_out_path = os.path.join(os.getcwd(), r'results',"model_epoch_{}.pth".format(epoch))
#     states = {
#         'epoch': epoch + 1,
#         'state_dict': model.state_dict(),
#         'optimizer': optimizer.state_dict(),
#     }
#     torch.save(states, model_out_path)
#     print("Checkpoint saved to {}".format(model_out_path))



if __name__ == '__main__':
    path = 0
    result_path = 0
    if torch.cuda.is_available():
        if os.name == 'nt':
            path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                r'OPPORTUNITY\OppSegBySubjectGesturesFull_113Validation.data')
            result_path = os.path.join(os.getcwd(), r'results')
        else:
            path = os.path.join(os.path.dirname(os.getcwd()), r'OPPORTUNITY/OppSegBySubjectGesturesFull_113Validation.data')
            result_path = os.path.join(os.getcwd(), r'results')
    else:
        path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), r'OPPORTUNITY\OppSegBySubjectGesturesFull_113Validation.data')

    # preprocess_data(dr,dp)

    print("Loading data...")

    if DATAFORMAT:

        Opp = OPPORTUNITY(path)
        X_train, y_train, X_validation, y_validation, X_test, y_test = Opp.load() #load_dataset(dp)
        assert NB_SENSOR_CHANNELS == X_train.shape[2]

        print(" ..after sliding window (training): inputs {0}, targets {1}".format(X_train.shape, y_train.shape))
        print(" ..after sliding window (validation): inputs {0}, targets {1}".format(X_validation.shape, y_validation.shape))
        print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))
        #
        # Data is reshaped since the input of the network is a 4 dimension tensor
        X_test = X_test.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
        X_test.astype(np.float32), y_test.reshape(len(y_test)).astype(np.uint8)

        X_train = X_train.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))  #inputs (46495, 1, 24, 113), targets (46495,)
        X_train.astype(np.float32), y_train.reshape(len(y_train)).astype(np.uint8)

        X_validation = X_validation.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
        X_validation.astype(np.float32), y_validation.reshape(len(y_validation)).astype(np.uint8)
        # X_train = X_train[1:300]
        # y_train = y_train[1:300]
        # X_test = X_test[1:300]
        # y_test = y_test[1:300]
        print(" after reshape: inputs {0}, targets {1}".format(X_train.shape, y_train.shape))
    else:
        Opp = OPPORTUNITY(path)
        X_train, y_train, X_validation, y_validation, X_test, y_test = Opp.load()  # load_dataset(dp)
        assert NB_SENSOR_CHANNELS_113 == X_train.shape[2]

        print(" ..after sliding window (training): inputs {0}, targets {1}".format(X_train.shape, y_train.shape))
        print(" ..after sliding window (validation): inputs {0}, targets {1}".format(X_validation.shape,
                                                                                     y_validation.shape))
        print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))
        #
        # Data is reshaped since the input of the network is a 4 dimension tensor
        X_test = X_test.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS_113))
        X_test.astype(np.float32), y_test.reshape(len(y_test)).astype(np.uint8)

        X_train = X_train.reshape(
            (-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS_113))  # inputs (46495, 1, 24, 113), targets (46495,)
        X_train.astype(np.float32), y_train.reshape(len(y_train)).astype(np.uint8)

        X_validation = X_validation.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS_113))
        X_validation.astype(np.float32), y_validation.reshape(len(y_validation)).astype(np.uint8)

    training_set = []
    validation_set = []
    testing_set = []


    if DATAFORMAT:
        if CONTAIN_NULLCLASS:
            X_train = list(X_train)
            y_train = list(y_train)

            for i in range(len(y_train)):
                x = X_train[i]
                y = y_train[i]
                x_object_channel = 0
                if y == 1 or 2 or 3 or 4:
                    x_object_channel = np.repeat(10,24).reshape(24, 1)
                elif y == 5 or 6:
                    x_object_channel = np.repeat(20,24).reshape(24, 1)
                elif y == 7 or 8:
                    x_object_channel = np.repeat(30,24).reshape(24, 1)
                elif y == 9 or 10 or 11 or 12 or 13 or 14:
                    x_object_channel = np.repeat(40,24).reshape(24, 1)
                elif y == 15:
                    x_object_channel = np.repeat(50,24).reshape(24, 1)
                elif y == 16:
                    x_object_channel = np.repeat(60,24).reshape(24, 1)
                elif y == 17:
                    x_object_channel = np.repeat(70,24).reshape(24, 1)
                else:
                    x_object_channel = np.repeat(0,24).reshape(24, 1)

                x_33_compass = x[-1, :, 33].reshape(24, 1)
                x = np.delete(x[-1, :, :], 33, 1)   # delete 33 col and append the last
                x = np.concatenate([x, x_33_compass,x_object_channel], axis=1)
                x = x.reshape(1,24,CHANNELS_OBJECT)

                xy = (x, y)
                training_set.append(xy)

            X_validation = list(X_validation)
            y_validation = list(y_validation)

            for i in range(len(y_validation)):
                x = X_validation[i]
                y = y_validation[i]
                x_object_channel = 0
                if y == 1 or 2 or 3 or 4:
                    x_object_channel = np.repeat(10, 24).reshape(24, 1)
                elif y == 5 or 6:
                    x_object_channel = np.repeat(20, 24).reshape(24, 1)
                elif y == 7 or 8:
                    x_object_channel = np.repeat(30, 24).reshape(24, 1)
                elif y == 9 or 10 or 11 or 12 or 13 or 14:
                    x_object_channel = np.repeat(40, 24).reshape(24, 1)
                elif y == 15:
                    x_object_channel = np.repeat(50, 24).reshape(24, 1)
                elif y == 16:
                    x_object_channel = np.repeat(60, 24).reshape(24, 1)
                elif y == 17:
                    x_object_channel = np.repeat(70, 24).reshape(24, 1)
                else:
                    x_object_channel = np.repeat(0, 24).reshape(24, 1)

                x_33_compass = x[-1, :, 33].reshape(24, 1)
                x = np.delete(x[-1, :, :], 33, 1)  # delete 33 col and append the last
                x = np.concatenate([x, x_33_compass, x_object_channel], axis=1)
                x = x.reshape(1, 24, CHANNELS_OBJECT)

                xy = (x, y)
                validation_set.append(xy)


            X_test = list(X_test)
            y_test = list(y_test)

            for j in range(len(y_test)):
                x = X_test[j]
                y = y_test[j]

                x_object_channel = 0
                if y == 1 or 2 or 3 or 4:
                    x_object_channel = np.repeat(10, 24).reshape(24, 1)
                elif y == 5 or 6:
                    x_object_channel = np.repeat(20, 24).reshape(24, 1)
                elif y == 7 or 8:
                    x_object_channel = np.repeat(30, 24).reshape(24, 1)
                elif y == 9 or 10 or 11 or 12 or 13 or 14:
                    x_object_channel = np.repeat(40, 24).reshape(24, 1)
                elif y == 15:
                    x_object_channel = np.repeat(50, 24).reshape(24, 1)
                elif y == 16:
                    x_object_channel = np.repeat(60, 24).reshape(24, 1)
                elif y == 17:
                    x_object_channel = np.repeat(70, 24).reshape(24, 1)
                else:
                    x_object_channel = np.repeat(0, 24).reshape(24, 1)

                x_33_compass = x[-1, :, 33].reshape(24, 1)
                x = np.delete(x[-1, :, :], 33, 1)  # delete 33 col and append the last
                x = np.concatenate([x, x_33_compass, x_object_channel], axis=1)
                x = x.reshape(1, 24, CHANNELS_OBJECT)

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
    else:
        X_train = list(X_train)
        y_train = list(y_train)

        for i in range(len(y_train)):
            x = X_train[i].astype(np.float32)
            y = y_train[i]
            xy = (x, y)
            training_set.append(xy)
        if DATA_AUGMENTATION:
            z_dimension = 100
            G = generator(z_dimension, 3872).cuda()  # generator model
            pre_train_path = os.path.join(os.getcwd(),
                                          r'dc_model\generator_16_epoch_300.pth')
            pretrain = torch.load(pre_train_path)
            G.load_state_dict(pretrain['state_dict'])

            z = Variable(torch.randn(500, z_dimension)).cuda()
            fake_x = G(z)
            for j in range(500):
                fx = fake_x[j].data.cpu().numpy().astype(np.float32)
                fy = np.array([16.0]).reshape(1)
                fake_xy = (fx,fy)
                training_set.append(fake_xy)

        X_validation = list(X_validation)
        y_validation = list(y_validation)

        for j in range(len(y_validation)):
            x = X_validation[j]
            y = y_validation[j]
            xy = (x, y)
            validation_set.append(xy)

        X_test = list(X_test)
        y_test = list(y_test)

        for k in range(len(y_test)):
            x = X_test[k]
            y = y_test[k]
            xy = (x, y)
            testing_set.append(xy)


    train_loader = DataLoader(dataset=training_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(dataset=validation_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=testing_set, batch_size=BATCH_SIZE, shuffle=True)
    train_logger = Logger(
        os.path.join(result_path, 'train.log'),
        ['epoch', 'loss', 'acc', 'lr', 'f1_score.avg'])
    train_batch_logger = Logger(
        os.path.join(result_path, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr', 'Failure_case_True', 'Failure_case_Pred'])
    val_logger = Logger(
        os.path.join(result_path, 'val.log'), ['epoch', 'loss', 'acc', 'f1_score.avg'])

    model = ConvLSTM()
    # If use CrossEntropyLoss，softmax wont be used in the final layer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=BASE_lr, momentum=0.9,weight_decay=0.0005)

    if torch.cuda.is_available():
        model.cuda()
        loss_function.cuda()
        print("Model on gpu")
    if pretrain_path:
        pre_train_path = os.path.join(os.getcwd(),
                                r'results\model_epoch_20_base.pth')
        pretrain = torch.load(pre_train_path)
        model.load_state_dict(pretrain['state_dict'])
        optimizer.load_state_dict(pretrain['optimizer'])
        print(model)


    train_correct = [0,]
    train_total = [0,]
    f1_train_total = AverageMeter()
    val_correct = [0,]
    val_total = [0,]
    f1_val_total = AverageMeter()

    # training and testing
    for epoch in range(EPOCH):
        train_epoch(epoch, train_loader, model, loss_function, optimizer,
                    train_logger, train_batch_logger, train_total, train_correct, f1_train_total)
        val_epoch(epoch, validation_loader, model, loss_function, val_logger, val_total, val_correct, f1_val_total)

    print('Accuracy of the Train model  {0} %, F1-score: {1}'.format(100 * train_correct[0] / train_total[0], f1_train_total.avg))
    print('Accuracy of the Validation model  {0} %, F1-score: {1}'.format(100 * val_correct[0] / val_total[0], f1_val_total.avg))


    # Test the model ####################################
    f1_test = AverageMeter()
    accuracies = AverageMeter()
    data_time = AverageMeter()
    end_time = time()
    model.eval()
    correct = 0
    total = 0

    for i, (seqs, labels) in enumerate(test_loader):
        # measure data loading time
        data_time.update(time() - end_time)

        seqs = get_variable(seqs.float())
        labels = get_variable(labels.long())

        outputs = model(seqs)
        labels = labels.squeeze()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        labels_reshape = labels
        correct += (predicted == labels_reshape.data).sum()

        ifCorrect = np.array((predicted == labels_reshape.data).cpu().numpy())
        failure_case_ind = np.where(ifCorrect == 0)
        label_for_failure_case = np.array(labels_reshape.cpu().numpy())
        label_for_pred_case = np.array(predicted.cpu().numpy())
        failure_case_True_label = label_for_failure_case[failure_case_ind]
        failure_case_Pred_label = label_for_pred_case[failure_case_ind]
        print('Failure_case_True  {0} % '.format(failure_case_True_label))
        print('Failure_case_Pred  {0} % '.format(failure_case_Pred_label))

        f1_test.update(f1_score(labels_reshape.cpu().numpy(), predicted.cpu().numpy(), average='micro'))
    print('Test Accuracy of the model  {0}%, F1-score {1}%'.format(100 * correct / total, f1_test.avg))

