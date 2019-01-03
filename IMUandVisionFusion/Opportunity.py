import torch
from torch.utils.data import Dataset, DataLoader
import pickle as cp
import numpy as np
import os

SUBJECT_NAME = ['S1-Drill','S1-ADL1','S1-ADL2','S1-ADL3','S1-ADL4','S1-ADL5', 'S2-Drill', 'S2-ADL1','S2-ADL2','S2-ADL3',
                                'S3-Drill','S3-ADL1', 'S3-ADL2','S3-ADL3','S2-ADL4', 'S2-ADL5','S3-ADL4', 'S3-ADL5']
SCENARIO = ['NULL', 'OpenDoor1', 'OpenDoor2', 'CloseDoor1', 'CloseDoor2', 'OpenFridge', 'CloseFridge', 'OpenDishwasher',
            'CloseDishwasher', 'OpenDrawer1', 'CloseDrawer1', 'OpenDrawer2', 'CloseDrawer2', 'OpenDrawer3',
            'CloseDrawer3', 'CleanTable', 'DrinkfromCup', 'ToggleSwitch']

class OPPORTUNITY(Dataset):

    def __init__(self, root, train=False):
        self.root = root
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.train = train
        self.load()


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


    def load(self):
        filename = self.root
        f = open(filename, 'rb')
        data = cp.load(f)
        f.close()

        training_set = data[0]
        testing_set = data[1]


        # np.savetxt('D:/Research/DeepConvLSTM/OPPORTUNITY/new.csv', y_train, delimiter=',')

        print(" ..from file {}".format(filename))
        print("Final datasets with size: | train {0} | test {1} | ".format(np.array(training_set).shape,
                                                                           np.array(testing_set).shape))
        # ..reading instances: train (557963, 113), test (118750, 113)

        for i in range(len(training_set)):
            for scen in SCENARIO:
                x = training_set[i].x[scen]
                self.X_train.append(x)
                y = training_set[i].y[scen]
                y = [item[0] for item in y]
                self.y_train.append(y)

        X = []
        for i in range(len(self.X_train)):
            if self.X_train[i] != []:
                for j in range(len(self.X_train[i])):
                    X.append(self.X_train[i][j])
        self.X_train = X

        Y = []
        for i in range(len(self.y_train)):
            if self.y_train[i] != []:
                for j in range(len(self.y_train[i])):
                    Y.append(self.y_train[i][j])
        self.y_train = Y


         ##############################
        for i in range(len(testing_set)):
            for scen in SCENARIO:
                x = testing_set[i].x[scen]
                self.X_test.append(x)
                y = testing_set[i].y[scen]
                y = [item[0] for item in y]
                self.y_test.append(y)

        X = []
        for i in range(len(self.X_test)):
            if self.X_test[i] != []:
                for j in range(len(self.X_test[i])):
                    X.append(self.X_test[i][j])
        self.X_test = X

        Y = []
        for i in range(len(self.y_test)):
            if self.y_test[i] != []:
                for j in range(len(self.y_test[i])):
                    Y.append(self.y_test[i][j])
        self.y_test = Y


if __name__ == '__main__':

    opp = OPPORTUNITY('D:/Research/DeepConvLSTM/OPPORTUNITY/gestures.data', train=True)
    # np.savetxt('D:/Research/DeepConvLSTM/OPPORTUNITY/new1.csv', np.array(opp.y_train), delimiter=',')
    x, y = opp[2000]
    a = 0

