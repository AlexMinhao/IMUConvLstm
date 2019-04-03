
import os
import zipfile
import argparse
import pickle as cp

from io import BytesIO
from pandas import Series
from definitions import *
from process_data_new import *
# Hardcoded names of the files defining the OPPORTUNITY challenge data. As named in the original data.
OPPORTUNITY_DATA_FILES = ['OpportunityUCIDataset/dataset/S1-Drill.dat',
                          'OpportunityUCIDataset/dataset/S1-ADL1.dat',
                          'OpportunityUCIDataset/dataset/S1-ADL2.dat',
                          'OpportunityUCIDataset/dataset/S1-ADL3.dat',
                          'OpportunityUCIDataset/dataset/S1-ADL4.dat',
                          'OpportunityUCIDataset/dataset/S1-ADL5.dat',
                          'OpportunityUCIDataset/dataset/S2-Drill.dat',
                          'OpportunityUCIDataset/dataset/S2-ADL1.dat',
                          'OpportunityUCIDataset/dataset/S2-ADL2.dat',
                          'OpportunityUCIDataset/dataset/S2-ADL3.dat',
                          'OpportunityUCIDataset/dataset/S3-Drill.dat',
                          'OpportunityUCIDataset/dataset/S3-ADL1.dat',
                          'OpportunityUCIDataset/dataset/S3-ADL2.dat',
                          'OpportunityUCIDataset/dataset/S3-ADL3.dat',
                          'OpportunityUCIDataset/dataset/S2-ADL4.dat',
                          'OpportunityUCIDataset/dataset/S2-ADL5.dat',
                          'OpportunityUCIDataset/dataset/S3-ADL4.dat',
                          'OpportunityUCIDataset/dataset/S3-ADL5.dat'
                          ]

# Hardcoded thresholds to define global maximums and minimums for every one of the 113 sensor channels employed in the
# OPPORTUNITY challenge
NORM_MAX_THRESHOLDS = [3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       250,    25,     200,    5000,   5000,   5000,   5000,   5000,   5000,
                       10000,  10000,  10000,  10000,  10000,  10000,  250,    250,    25,
                       200,    5000,   5000,   5000,   5000,   5000,   5000,   10000,  10000,
                       10000,  10000,  10000,  10000,  250, ]

NORM_MIN_THRESHOLDS = [-3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -250,   -100,   -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,
                       -10000, -10000, -10000, -10000, -10000, -10000, -250,   -250,   -100,
                       -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,  -10000, -10000,
                       -10000, -10000, -10000, -10000, -250, ]

NORM_MAX_FourSENSOR = [3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       250,    25,     200,    5000,   5000,   5000,   5000,   5000,   5000,
                       10000,  10000,  10000,  10000,  10000,  10000,  250,    250,    25,
                       200,    5000,   5000,   5000,   5000,   5000,   5000,   10000,  10000,
                       10000,  10000,  10000,  10000,  250, ]

NORM_MIN_FourSENSOR = [-3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                       -250,   -100,   -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,
                       -10000, -10000, -10000, -10000, -10000, -10000, -250,   -250,   -100,
                       -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,  -10000, -10000,
                       -10000, -10000, -10000, -10000, -250, ]


def select_columns_opp(data):
    """Selection of the 113 columns employed in the OPPORTUNITY challenge

    :param data: numpy integer matrix
        Sensor data (all features)
    :return: numpy integer matrix
        Selection of features
    """

    #                     included-excluded
    features_delete = np.arange(46, 50)                                            # [46,47,48,49]
    features_delete = np.concatenate([features_delete, np.arange(59, 63)])         # [46, 47, 48, 49, 59, 60, 61, 62]
    features_delete = np.concatenate([features_delete, np.arange(72, 76)])         # [46, 47, 48, 49, 59, 60, 61, 62, 72, 73, 74, 75]
    features_delete = np.concatenate([features_delete, np.arange(85, 89)])         # [46, 47, 48, 49, 59, 60, 61, 62, 72, 73, 74, 75, 85, 86, 87, 88]
    features_delete = np.concatenate([features_delete, np.arange(98, 102)])        # [ 46,  47,  48,  49,  59,  60,  61,  62,  72,  73,  74,  75,  85, 86,  87,  88,  98,  99, 100, 101]
    features_delete = np.concatenate([features_delete, np.arange(134, 243)])       # [ 46,  47,  48,  49,  59,  60,  61,  62,  72,  73,  74,  75,  85, 86,  87,  88,  98,  99, 100, 101, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242]
    features_delete = np.concatenate([features_delete, np.arange(244, 249)])       #

    # # reduce the num of sensors
    # features_delete = np.arange(1, 63)  # [46,47,48,49]
    # features_delete = np.concatenate([features_delete, np.arange(72, 89)])  # [46, 47, 48, 49, 59, 60, 61, 62, 72, 73, 74, 75]
    # features_delete = np.concatenate([features_delete, np.arange(98, 102)])  # [ 46,  47,  48,  49,  59,  60,  61,  62,  72,  73,  74,  75,  85, 86,  87,  88,  98,  99, 100, 101]
    # features_delete = np.concatenate([features_delete, np.arange(134, 243)])  # [ 46,  47,  48,  49,  59,  60,  61,  62,  72,  73,  74,  75,  85, 86,  87,  88,  98,  99, 100, 101, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242]
    # features_delete = np.concatenate([features_delete, np.arange(244, 249)])  #
    # temp = np.delete(data, features_delete, 1)

    return np.delete(data, features_delete, 1)


def normalize(data, max_list, min_list):
    """Normalizes all sensor channels

    :param data: numpy integer matrix
        Sensor data
    :param max_list: numpy integer array
        Array containing maximums values for every one of the 113 sensor channels
    :param min_list: numpy integer array
        Array containing minimum values for every one of the 113 sensor channels
    :return:
        Normalized sensor data
    """
    max_list, min_list = np.array(max_list), np.array(min_list)
    diffs = max_list - min_list
    for i in np.arange(data.shape[1]):
        data[:, i] = (data[:, i]-min_list[i])/diffs[i]
    #     Checking the boundaries
    data[data > 1] = 0.99
    data[data < 0] = 0.00
    return data


def divide_x_y(data, label):
    """Segments each sample into features and label

    :param data: numpy integer matrix
        Sensor data
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numpy integer array
        Features encapsulated into a matrix and labels as an array
    """

    data_x = data[:, 1:114]
    if label not in ['locomotion', 'gestures']:
            raise RuntimeError("Invalid label: '%s'" % label)
    if label == 'locomotion':
        data_y = data[:, 114]  # Locomotion label
    elif label == 'gestures':
        data_y = data[:, 115]  # Gestures label

    # Reduce the num of sensors
    # data_x = data[:, 1:51]
    # if label not in ['locomotion', 'gestures']:
    #         raise RuntimeError("Invalid label: '%s'" % label)
    # if label == 'locomotion':
    #     data_y = data[:, 51]  # Locomotion label
    # elif label == 'gestures':
    #     data_y = data[:, 52]  # Gestures label

    return data_x, data_y


def adjust_idx_labels(data_y, label):
    """Transforms original labels into the range [0, nb_labels-1]

    :param data_y: numpy integer array
        Sensor labels
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer array
        Modified sensor labels
    """

    if label == 'locomotion':  # Labels for locomotion are adjusted
        data_y[data_y == 4] = 3
        data_y[data_y == 5] = 4
    elif label == 'gestures':  # Labels for gestures are adjusted
        data_y[data_y == 406516] = 1
        data_y[data_y == 406517] = 2
        data_y[data_y == 404516] = 3
        data_y[data_y == 404517] = 4
        data_y[data_y == 406520] = 5
        data_y[data_y == 404520] = 6
        data_y[data_y == 406505] = 7
        data_y[data_y == 404505] = 8
        data_y[data_y == 406519] = 9
        data_y[data_y == 404519] = 10
        data_y[data_y == 406511] = 11
        data_y[data_y == 404511] = 12
        data_y[data_y == 406508] = 13
        data_y[data_y == 404508] = 14
        data_y[data_y == 408512] = 15
        data_y[data_y == 407521] = 16
        data_y[data_y == 405506] = 17
    return data_y


def check_data(data_set):
    """Try to access to the file and checks if dataset is in the data directory
       In case the file is not found try to download it from original location

    :param data_set:
            Path with original OPPORTUNITY zip file
    :return:
    """
    print('Checking dataset {0}'.format(data_set))
    data_dir, data_file = os.path.split(data_set)
    # When a directory is not provided, check if dataset is in the data directory
    if data_dir == "" and not os.path.isfile(data_set):
        new_path = os.path.join(os.path.split(__file__)[0], "data", data_set)
        if os.path.isfile(new_path) or data_file == 'OpportunityUCIDataset.zip':
            data_set = new_path

    # When dataset not found, try to download it from UCI repository
    if (not os.path.isfile(data_set)) and data_file == 'OpportunityUCIDataset.zip':
        print('... dataset path {0} not found'.format(data_set))
        import urllib
        origin = (
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip'
        )
        if not os.path.exists(data_dir):
            print('... creating directory {0}'.format(data_dir))
            os.makedirs(data_dir)
        print('... downloading data from {0}'.format(origin))
        urllib.urlretrieve(origin, data_set)

    return data_dir


def process_dataset_file(data, label):
    """Function defined as a pipeline to process individual OPPORTUNITY files

    :param data: numpy integer matrix
        Matrix containing data samples (rows) for every sensor channel (column)
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into features (x) and labels (y)
    """

    # Select correct columns
    data = select_columns_opp(data)

    # Colums are segmentd into features and labels
    data_x, data_y = divide_x_y(data, label)
    data_y = adjust_idx_labels(data_y, label)   #label: string, ['gestures' (default), 'locomotion']
    data_y = data_y.astype(int)

    # Perform linear interpolation
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T

    # Remaining missing data are converted to zero
    data_x[np.isnan(data_x)] = 0

    # All sensor channels are normalized
    data_x = normalize(data_x, NORM_MAX_THRESHOLDS, NORM_MIN_THRESHOLDS)

    return data_x, data_y


def generate_data(dataset, target_filename, label):
    """Function to read the OPPORTUNITY challenge raw data and process all sensor channels

    :param dataset: string
        Path with original OPPORTUNITY zip file
    :param target_filename: string
        Processed file
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized. The OPPORTUNITY dataset includes several annotations to perform
        recognition modes of locomotion/postures and recognition of sporadic gestures.
    """

    data_dir = check_data(dataset)

    data_x = np.empty((0, NB_SENSOR_CHANNELS_113))
    data_y = np.empty(0)

    zf = zipfile.ZipFile(dataset)

    subject = SUBJECT
    print('Processing dataset files ...')
    for filename in OPPORTUNITY_DATA_FILES:
        try:
            data = np.loadtxt(BytesIO(zf.read(filename)))
            print('... file {0}'.format(filename))
            x, y = process_dataset_file(data, label)

            s = Subject(FILES_MAP_SUB[filename], x, y)
            scenario = s.divide_scenario()
            s.divide_data_label()
            subject[FILES_MAP_SUB[filename]] = s
            length = subject[FILES_MAP_SUB[filename]].__len__()
            print("Subject: {0} Length: {1} x_len: {2}".format(filename, length, len(x)))
            data_x = np.vstack((data_x, x))
            data_y = np.concatenate([data_y, y])
        except KeyError:
            print('ERROR: Did not find {0} in zip file'.format(filename))

    # Dataset is segmented into train and test
    # nb_training_samples = 557963
    # # The first 18 OPPORTUNITY data files define the traning dataset, comprising 557963 samples
    # X_train, y_train = data_x[:nb_training_samples, :], data_y[:nb_training_samples]
    # X_test, y_test = data_x[nb_training_samples:, :], data_y[nb_training_samples:]
    print("Number of X_train: {0} X_test: {1}" .format(len(data_x[:557963, :]), len(data_x[557963:, :])))
                            #51455               #
    training_set = [subject['S1-Drill'], subject['S1-ADL1'], subject['S1-ADL2'], subject['S1-ADL3'], subject['S1-ADL4'], subject['S1-ADL5'],
               subject['S2-Drill'], subject['S2-ADL1'], subject['S2-ADL2'], subject['S3-Drill'], subject['S3-ADL1'], subject['S3-ADL2']]
    validation_set = [subject['S2-ADL3'], subject['S3-ADL3']]
    testing_set = [subject['S2-ADL4'], subject['S2-ADL5'], subject['S3-ADL4'], subject['S3-ADL5']]
    # print("Final datasets with size: | train {0} | test {1} | ".format(X_train.shape, X_test.shape))

    print("Final datasets with size: | train {0} | validation {1} |test {2} | ".format(np.array(training_set).shape, np.array(validation_set).shape, np.array(testing_set).shape))

    obj = [training_set, validation_set, testing_set]
    f = open(os.path.join(data_dir, target_filename), 'wb')
    cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
    f.close()


def get_args():
    '''This function parses and return arguments passed in'''
    parser = argparse.ArgumentParser(
        description='Preprocess OPPORTUNITY dataset')
    # Add arguments
    parser.add_argument(
        '-i', '--input', type=str, help='OPPORTUNITY zip file', required=True)
    parser.add_argument(
        '-o', '--output', type=str, help='Processed data file', required=True)
    parser.add_argument(
        '-t', '--task', type=str.lower, help='Type of activities to be recognized', default="gestures", choices = ["gestures", "locomotion"], required=False)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    dataset = args.input
    target_filename = args.output
    label = args.task      #label: string, ['gestures' (default), 'locomotion']
    # Return all variable values
    return dataset, target_filename, label

if __name__ == '__main__':

    # OpportunityUCIDataset_zip, output, l = get_args()
    #generate_data('C:/ALEX/Doc/paper/PytorchTuto/OPPORTUNITY/OpportunityUCIDataset.zip', 'C:/ALEX/Doc/paper/PytorchTuto/OPPORTUNITY/OppSegBySubjectGesturesReduceSensors.data', 'gestures')
    # generate_data('C:/ALEX/Doc/paper/PytorchTuto/OPPORTUNITY/OpportunityUCIDataset.zip',
    #               'C:/ALEX/Doc/paper/PytorchTuto/OPPORTUNITY/OppSegBySubjectGesturesReduceSensorsValidation.data', 'gestures')
    generate_data('C:/ALEX/Doc/paper/PytorchTuto/OPPORTUNITY/OpportunityUCIDataset.zip',
                  'C:/ALEX/Doc/paper/PytorchTuto/OPPORTUNITY/OppSegBySubjectGesturesFull_113Validation.data',
                  'gestures')

