import torch
from torch.utils.data import Dataset, DataLoader
import pickle as cp
import numpy as np
import os
from sliding_window import sliding_window
from definitions import *

SUBJECT_NAME = ['S1-Drill','S1-ADL1','S1-ADL2','S1-ADL3','S1-ADL4','S1-ADL5', 'S2-Drill', 'S2-ADL1','S2-ADL2','S2-ADL3',
                                'S3-Drill','S3-ADL1', 'S3-ADL2','S3-ADL3','S2-ADL4', 'S2-ADL5','S3-ADL4', 'S3-ADL5']
SCENARIO = ['NULL', 'OpenDoor1', 'OpenDoor2', 'CloseDoor1', 'CloseDoor2', 'OpenFridge', 'CloseFridge', 'OpenDishwasher',
            'CloseDishwasher', 'OpenDrawer1', 'CloseDrawer1', 'OpenDrawer2', 'CloseDrawer2', 'OpenDrawer3',
            'CloseDrawer3', 'CleanTable', 'DrinkfromCup', 'ToggleSwitch']
DATASET = {'Opportunity': 0, 'PAMAP2': 1}


class ImportDataSet:

    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.load()

    def load(self):
        print(" Load dataset ...")
        index = DATASET[self.name]
        if index == 0:
            dataset = OPPORTUNITY(self.path)
            return dataset


class OPPORTUNITY(Dataset):

    def __init__(self, root, train=False):
        self.root = root
        self.X_train = []
        self.y_train = []
        self.X_validation = []
        self.y_validation = []
        self.X_test = []
        self.y_test = []
        self.train = train
        self.data_process()

    def __getitem__(self, index):
        if self.train:
            data, label = self.X_train[index], self.y_train[index]
            return data, label
        else:
            data, label = self.X_test[index], self.y_test[index]
            return data, label

    def __len__(self):
        if self.train:
            return len(self.X_train)
        else:
            return len(self.X_test)

    def concatenate_same_class(self, data_x, data_y):
        X = np.zeros((1, 24, NB_SENSOR_CHANNELS_113))
        Y = np.zeros((1,1))
        for i in range(len(data_x)):
            if len(data_x[i]) > 0:
                # Padding
                padding_size = SLIDING_WINDOW_STEP - len(data_x[i]) % SLIDING_WINDOW_STEP \
                    if len(data_x[i]) > SLIDING_WINDOW_LENGTH else SLIDING_WINDOW_LENGTH+SLIDING_WINDOW_STEP-len(data_x[i])
                padding_x = np.zeros((padding_size, NB_SENSOR_CHANNELS_113))
                data_x[i] = np.row_stack((data_x[i], padding_x))
                data_y[i] = data_y[i].repeat(len(data_x[i]))
                x_temp = sliding_window(data_x[i], (SLIDING_WINDOW_LENGTH, data_x[i].shape[1]), (SLIDING_WINDOW_STEP, 1))
                y_temp = sliding_window(data_y[i], SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
                y_temp = np.asarray([[i[-1]] for i in y_temp])
                x_temp.astype(np.float32), y_temp.reshape(len(y_temp)).astype(np.uint8)
                X = np.row_stack((X, x_temp))
                Y = np.row_stack((Y, y_temp))
        return X, Y

    def data_process(self):
        print(" Dataset Process ...")
        filename = self.root
        f = open(filename, 'rb')
        data = cp.load(f)
        f.close()

        training_set = data[0]
        validation_set = data[1]
        testing_set = data[2]

        print(" ..from file {}".format(filename))
        print("Final datasets with size: | train {0} | validation {1} |test {2} | ".format(np.array(training_set).shape,
                                                                                           np.array(
                                                                                               validation_set).shape,
                                                                                           np.array(testing_set).shape))

        # ..reading instances: train (557963, 113), test (118750, 113)
        # (46495,24,113)  (46495,)  column
        x_train = []
        y_train = []
        for i in range(len(training_set)):
            for scen in SCENARIO:
                x = training_set[i].x[scen]
                #self.X_train.append(x)
                y = training_set[i].y[scen]
                y = [item[0] for item in y]
                #self.y_train.append(y)

                x_scen, y_scen = self.concatenate_same_class(x, y)
                x_train.append(x_scen[1:])
                y_train.append(y_scen[1:])
        self.X_train = np.row_stack(x_train)
        self.y_train = np.row_stack(y_train)

        x_validation = []
        y_validation = []
        for i in range(len(validation_set)):
            for scen in SCENARIO:
                x = validation_set[i].x[scen]
                #self.X_train.append(x)
                y = validation_set[i].y[scen]
                y = [item[0] for item in y]
                #self.y_train.append(y)

                x_scen, y_scen = self.concatenate_same_class(x, y)
                x_validation.append(x_scen[1:])
                y_validation.append(y_scen[1:])
        self.X_validation = np.row_stack(x_validation)
        self.y_validation = np.row_stack(y_validation)

        x_test = []
        y_test = []
        for i in range(len(testing_set)):
            for scen in SCENARIO:
                x = testing_set[i].x[scen]
                # self.X_train.append(x)
                y = testing_set[i].y[scen]
                y = [item[0] for item in y]
                # self.y_train.append(y)

                x_scen, y_scen = self.concatenate_same_class(x, y)
                x_test.append(x_scen[1:])
                y_test.append(y_scen[1:])
        self.X_test = np.row_stack(x_test)
        self.y_test = np.row_stack(y_test)
        print("Final datasets with size: | train {0} | test {1} | ".format(np.array(self.X_train).shape,
                                                                           np.array(self.X_test).shape))

    def load(self):
        return self.X_train, self.y_train, self.X_validation, self.y_validation, self.X_test, self.y_test
        # X = []
        # for i in range(len(self.X_train)):
        #     if self.X_train[i] != []:
        #         for j in range(len(self.X_train[i])):
        #             X.append(self.X_train[i][j])
        # self.X_train = X
        #
        # Y = []
        # for i in range(len(self.y_train)):
        #     if self.y_train[i] != []:
        #         for j in range(len(self.y_train[i])):
        #             Y.append(self.y_train[i][j])
        # self.y_train = Y

        # for i in range(len(testing_set)):
        #     for scen in SCENARIO:
        #         x = testing_set[i].x[scen]
        #         self.X_test.append(x)
        #         y = testing_set[i].y[scen]
        #         y = [item[0] for item in y]
        #         self.y_test.append(y)
        #
        # X = []
        # for i in range(len(self.X_test)):
        #     if self.X_test[i] != []:
        #         for j in range(len(self.X_test[i])):
        #             X.append(self.X_test[i][j])
        # self.X_test = X
        #
        # Y = []
        # for i in range(len(self.y_test)):
        #     if self.y_test[i] != []:
        #         for j in range(len(self.y_test[i])):
        #             Y.append(self.y_test[i][j])
        # self.y_test = Y




if __name__ == '__main__':

    path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), r'OPPORTUNITY\OppSegBySubjectGestures.data')
    # opp = ImportDataSet('Opportunity', path)
    opp = OPPORTUNITY(path)
    a = 0

