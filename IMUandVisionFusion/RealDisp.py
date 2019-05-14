import os
from scipy import io
import numpy as np
import pandas as pd
from scipy import stats
import pickle


# REALDISP Activity Recognition Dataset Data Set


def sliding_window(dataset, window_size, overlay, num):
    # delete frame + timestamp -> shouldnt be important for classification
    # only raw data remains
    activity = dataset[:, 119]
    dataset = np.delete(dataset, [0, 1, 119], 1)

    for i in range(0, dataset.shape[0] - window_size, window_size - overlay):

        # sliding window action
        temp_data = dataset[i:i + window_size, :]
        temp_activity = activity[i:i + window_size]

        if i == 0:
            activity_data = np.array([stats.mode(temp_activity)[0]])
            data = temp_data.reshape(1, temp_data.shape[0] * temp_data.shape[1])

        else:
            activity_data = np.concatenate((activity_data, np.array([stats.mode(temp_activity)[0]])), axis=0)
            data = np.concatenate((data, temp_data.reshape(1, temp_data.shape[0] * temp_data.shape[1])), axis=0)

        del temp_data
        del temp_activity

    del dataset

    data = np.concatenate((activity_data, np.repeat([[num]], data.shape[0], axis=0), data), axis=1)

    return data


def load_set(window_size, overlay):
    print('Processing Data')

    file_names = ['ideal', 'self', 'mutual4', 'mutual5', 'mutual6', 'mutual7']
    root = 'C:\ALEX\Doc\paper\PytorchTuto'
    # subjects (17)
    for i in range(1, 18):

        print('Subject ' + str(i))

        # different trials (max 6)
        for j in range(0, 6):
            root = os.path.dirname(os.path.dirname(os.getcwd()))
            p = 'REALDISP\\realistic_sensor_displacement\\subject' + str(i) + '_' + file_names[j] + '.log'
            path = os.path.join(root, p)
            if not os.path.exists(path):
                continue

            raw_data = pd.read_csv(path, sep='\t', header=None)

            # first of all delete the missing values:
            raw_data.replace(['na', 'nan', 'NaN', 'NaT', 'inf', '-inf', 'nan', '?'], np.nan, inplace=True)
            raw_data = raw_data.dropna().values.astype(float)

            if j == 0:
                data = sliding_window(raw_data, window_size, overlay, j)
            else:
                data = np.concatenate((data, sliding_window(raw_data, window_size, overlay, j)), axis=0)
            del raw_data

        #### manage labels
        data = np.concatenate((np.repeat([[i]], data.shape[0], axis=0), data), axis=1)

        #### DUMP to pickle !!!!!
        path = root
        pickle.dump(data, open(path + 'Subject' + str(i) + '_preprocessed.txt', 'wb'), pickle.HIGHEST_PROTOCOL)

        print(data.shape)
        del data
        print('-------------------------------------------------------------------')

    return 1


# TODO -> change the path!!
def merge_subjects(window_size):
    path = 'C:/ALEX/Doc/paper/PytorchTuto/REALDISP/'

    # attribute names
    sensor_names = ['_LC', '_LT', '_RC', '_RT', '_BACK', '_LLA', '_LUA', '_RLA', '_RUA']
    columns = ['Activity', 'Trial']
    sensors = ['_AccX', '_AccY', '_AccZ', '_GyrX', '_GyrY', '_GyrZ', '_MagX', '_MagY', '_MagZ',
               '_Q1', '_Q2', '_Q3', '_Q4']
    att_names = [str(k) + j + i for k in range(1, window_size + 1) for j in sensor_names for i in sensors]
    columns.extend(att_names)

    # merge all subjects
    for i in range(1, 18):
        if i == 1:
            data = pickle.load(open(path + 'Subject' + str(i) + '_preprocessed.txt', 'rb'))
        else:
            data = np.concatenate((data, pickle.load(open(path + 'Subject' + str(i) + '_preprocessed.txt', 'rb'))),
                                  axis=0)

    # make a dataframe
    data = pd.DataFrame(data[:, 1:], columns=columns, index=data[:, 0])
    data.index.name = 'Subject'

    # export dataframe
    pickle.dump(data, open('REALDISP/realdisp_full_preprocessed.txt', 'wb'), pickle.HIGHEST_PROTOCOL)

    print(data.shape)
    del data

    return 1


def preprocess(window_size, overlay):
    if window_size <= overlay:
        print("Error: Window size and/or overlay")
        exit()

    if os.path.exists('REALDISP/realistic_sensor_displacement'):
        load_set(window_size, overlay)
        merge_subjects(window_size)
        return 1

    else:
        print("The specified dataset is not available/doesnt exsist")


if __name__ == '__main__':
    load_set(24, 12)


