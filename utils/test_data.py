import pandas as pd
import os
import numpy as np

def readData(file_name):
    init_data = pd.read_csv(file_name, usecols=[2, 3, 4, 5])
    return init_data.values, init_data.columns.tolist()


if __name__ == '__main__':
    file1 = './dataTest/train_lstm.csv'
    file2 = './dataTest/val_lstm.csv'
    file3 = './dataTest/test_lstm.csv'

    train_data, train_column_name = readData(file1)
    val_data, val_column_name = readData(file2)
    test_data, test_column_name = readData(file3)

    whole_data = train_data
    whole_data = np.vstack((whole_data, val_data))
    whole_data = np.vstack((whole_data, test_data))

    mean = np.mean(whole_data, axis=0)
    std = np.std(whole_data, axis=0)

    print("Mean is: {}, Std is:{}".format(mean, std))
    print(whole_data[:20, [0, 1]])
    
