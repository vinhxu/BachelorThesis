
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from dataGenerator import roomsDimension

def shuffle_csv_data(path = './01_rawData/rbRoute2Data_10k.csv', after_shuffled_file_name = './02_processedData/pbRoute2Data_10k.csv'):

    data = pd.read_csv(path)
    #  Shuffle data using sample and reset to new index
    data = data.sample(frac=1).reset_index(drop=True)

    data.to_csv(after_shuffled_file_name, index=False)
    return after_shuffled_file_name

# shuffle_csv_data()


def scaler_min_max(data,feature_range=(0,1)):
    scaler = MinMaxScaler(feature_range= feature_range)
    scaler.fit(data)
    data = scaler.transform(data)
    return data, scaler


def convert_to_oneHot(original_array):
    # return original_array
    nb_classes = int(original_array.max()+1)
    original_array.reshape(-1)
    converted_array = np.eye(nb_classes)[original_array.astype(int)]
    return converted_array

def load_data(mode='train', oneHot=True):
    """
    Function to load correct data set
    :param mode: train or test
    :return: data set with XY-coordinates and correct labels
    """

    if mode == 'train':
        x_train, y_train, x_valid, y_valid = points_train_coordinates, points_train_labels, \
                                             points_valid_coordinates, points_valid_labels
        if (oneHot==True):
            y_train = convert_to_oneHot(y_train)
            y_valid = convert_to_oneHot(y_valid)

        return x_train, y_train, x_valid, y_valid

    elif mode == 'test':
        x_test, y_test = points_test_coordinates, points_test_labels
        if (oneHot==True):
            y_test = convert_to_oneHot(y_test)

        return x_test, y_test

data = pd.read_csv('./02_processedData/pbRoute1Data_10k.csv').values
data, scaler = scaler_min_max(data, feature_range = (0,1))

# Train, Validation and Test data, Train: 79%, Validation 7%, Test: 14%
# Train: 79%
train_start = 0
train_end   = int(np.floor(0.79*len(data)))
data_train  = data[np.arange(train_start, train_end), :]
# Validation 7%
valid_start = train_end + 1
valid_end   = int(np.floor((0.79+0.07)*len(data)))
data_valid  = data[np.arange(valid_start, valid_end), :]
# Test: 14%
test_start  = valid_end + 1
test_end    = len(data)
data_test   = data[np.arange(test_start, test_end), :]


# # Build input (x,y) and output (label)
points_train_coordinates  = data_train[:, 1:]
points_train_labels       = data_train[:, 0]
points_valid_coordinates  = data_valid[:, 1:]
points_valid_labels       = data_valid[:, 0]
points_test_coordinates   = data_test[:, 1:]
points_test_labels        = data_test[:, 0]



walkingPathData = pd.read_csv('./01_rawData/walkingPathData.csv').values

walkingPathData = np.append(walkingPathData, [[0, roomsDimension.min().values[0], roomsDimension.min().values[1]]], axis=0)
walkingPathData = np.append(walkingPathData, [[0, roomsDimension.max().values[0], roomsDimension.max().values[1]]], axis=0)

walkingPathData = scaler.transform(walkingPathData)

