from helper import *
import os
import zipfile
import argparse
import pickle as cp

from io import BytesIO
from pandas import Series
# Hardcoded names of the files defining the OPPORTUNITY challenge data. As named in the original data.
SCENARIO = ['NULL', 'OpenDoor1', 'OpenDoor2', 'CloseDoor1', 'CloseDoor2', 'OpenFridge', 'CloseFridge', 'OpenDishwasher',
            'CloseDishwasher', 'OpenDrawer1', 'CloseDrawer1', 'OpenDrawer2', 'CloseDrawer2', 'OpenDrawer3',
            'CloseDrawer3', 'CleanTable', 'DrinkfromCup', 'ToggleSwitch']

FILES_MAP_SUB = {       'OpportunityUCIDataset/dataset/S1-Drill.dat': 'S1-Drill',
                          'OpportunityUCIDataset/dataset/S1-ADL1.dat': 'S1-ADL1',
                          'OpportunityUCIDataset/dataset/S1-ADL2.dat': 'S1-ADL2',
                          'OpportunityUCIDataset/dataset/S1-ADL3.dat': 'S1-ADL3',
                          'OpportunityUCIDataset/dataset/S1-ADL4.dat': 'S1-ADL4',
                          'OpportunityUCIDataset/dataset/S1-ADL5.dat': 'S1-ADL5',
                          'OpportunityUCIDataset/dataset/S2-Drill.dat':'S2-Drill',
                          'OpportunityUCIDataset/dataset/S2-ADL1.dat': 'S2-ADL1',
                          'OpportunityUCIDataset/dataset/S2-ADL2.dat': 'S2-ADL2',
                          'OpportunityUCIDataset/dataset/S2-ADL3.dat': 'S2-ADL3',
                          'OpportunityUCIDataset/dataset/S3-Drill.dat':'S3-Drill',
                          'OpportunityUCIDataset/dataset/S3-ADL1.dat': 'S3-ADL1',
                          'OpportunityUCIDataset/dataset/S3-ADL2.dat': 'S3-ADL2',
                          'OpportunityUCIDataset/dataset/S3-ADL3.dat': 'S3-ADL3',
                          'OpportunityUCIDataset/dataset/S2-ADL4.dat': 'S2-ADL4',
                          'OpportunityUCIDataset/dataset/S2-ADL5.dat': 'S2-ADL5',
                          'OpportunityUCIDataset/dataset/S3-ADL4.dat': 'S3-ADL4',
                          'OpportunityUCIDataset/dataset/S3-ADL5.dat': 'S3-ADL5'}
SUBJECT = {'S1-Drill': 0, #T
           'S1-ADL1': 1,  #T
           'S1-ADL2': 2,  #T
           'S1-ADL3': 3,  #T
           'S1-ADL4': 4,  #T
           'S1-ADL5': 5,  #T
           'S2-Drill':6,  #T
           'S2-ADL1': 7,  #T
           'S2-ADL2': 8,  #T
           'S2-ADL3': 9,  #V
           'S3-Drill':10,  #T
           'S3-ADL1': 11,  #T
           'S3-ADL2': 12,  #T
           'S3-ADL3': 13,  #V
           'S2-ADL4': 14,
           'S2-ADL5': 15,
           'S3-ADL4': 16,
           'S3-ADL5': 17}
CLASS = {   SCENARIO[0]:            0,
            SCENARIO[1]:            1,
            SCENARIO[2]:            2,
            SCENARIO[3]:            3,
            SCENARIO[4]:            4,
            SCENARIO[5]:            5,
            SCENARIO[6]:            6,
            SCENARIO[7]:            7,
            SCENARIO[8]:            8,
            SCENARIO[9]:            9,
            SCENARIO[10]:          10,
            SCENARIO[11]:          11,
            SCENARIO[12]:          12,
            SCENARIO[13]:          13,
            SCENARIO[14]:          14,
            SCENARIO[15]:          15,
            SCENARIO[16]:          16,
            SCENARIO[17]:          17}


class Subject(object):

    def __init__(self, name, data_x, data_y):
        self.name = name
        self.data_x = data_x
        self.data_y = data_y
        self.data_length = 0
        self.scenario = {'NULL': [], 'OpenDoor1': [], 'OpenDoor2': [],
                         'CloseDoor1': [], 'CloseDoor2': [], 'OpenFridge': [],
                         'CloseFridge': [], 'OpenDishwasher': [], 'CloseDishwasher': [],
                         'OpenDrawer1': [], 'CloseDrawer1': [], 'OpenDrawer2': [], 'CloseDrawer2': [],
                         'OpenDrawer3': [], 'CloseDrawer3': [], 'CleanTable': [], 'DrinkfromCup': [], 'ToggleSwitch': []}
        self.x =  {'NULL': [], 'OpenDoor1': [], 'OpenDoor2': [],
                         'CloseDoor1': [], 'CloseDoor2': [], 'OpenFridge': [],
                         'CloseFridge': [], 'OpenDishwasher': [], 'CloseDishwasher': [],
                         'OpenDrawer1': [], 'CloseDrawer1': [], 'OpenDrawer2': [], 'CloseDrawer2': [],
                         'OpenDrawer3': [], 'CloseDrawer3': [], 'CleanTable': [], 'DrinkfromCup': [], 'ToggleSwitch': []}
        self.y =  {'NULL': [], 'OpenDoor1': [], 'OpenDoor2': [],
                         'CloseDoor1': [], 'CloseDoor2': [], 'OpenFridge': [],
                         'CloseFridge': [], 'OpenDishwasher': [], 'CloseDishwasher': [],
                         'OpenDrawer1': [], 'CloseDrawer1': [], 'OpenDrawer2': [], 'CloseDrawer2': [],
                         'OpenDrawer3': [], 'CloseDrawer3': [], 'CleanTable': [], 'DrinkfromCup': [], 'ToggleSwitch': []}

    def __len__(self):
        for scene in SCENARIO:
            for i in range(len(self.x[scene])):
                self.data_length = self.data_length+len(self.x[scene][i])
        return self.data_length


    def divide_scenario(self):
        length = 0
        for scene in SCENARIO:
            index = np.argwhere(self.data_y == CLASS[scene])
            length = length + len(index)
            self.seperate(index, scene)
        a = 0
        return self.scenario

    def seperate(self, index, scenario):
        count = 1
        ind = 0
        for i in range(len(index) - 1):
            if index[i + 1] - index[i] != 1:
                count = count + 1
                motionIndex = index[ind:i+1]  # one more word
                self.scenario[scenario].append(motionIndex)
                ind = i + 1
            if i == len(index) - 2:
                motionIndex = index[ind:]
                self.scenario[scenario].append(motionIndex)

    def divide_data_label(self):
        data_length = 0
        for scene in SCENARIO:
            motion = []
            label = []
            for i in range(len(self.scenario[scene])):
                index = self.scenario[scene][i]
                if self.scenario[scene][i].size:
                    index = list(np.concatenate(self.scenario[scene][i]))
                #print('IndexLength:{0}'.format(len(index)))
                m_seg = self.data_x[index, :]
                l_seg = self.data_y[index]

                motion.append(m_seg)
                label.append(l_seg)
            self.x[scene] = motion
            self.y[scene] = label

        for scene in SCENARIO:
            for i in range(len(self.x[scene])):
                data_length = data_length + len(self.x[scene][i])

        a = 0





