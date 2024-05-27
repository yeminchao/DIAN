import numpy as np
import scipy.io as sio
import random

# load data

a = []
for i in range(0, y_target.shape[0]):
    for j in range(0, y_target.shape[1]):
        if y_target[i][j] != 0:
            a.append(i * 400 + j)

y_target = y_target.ravel()
y_target = y_target[y_target > 0]
y_target -= 1


source_train_number = 200
target_train_number = 5
class_num = -1
for label in np.unique(y_target):
    class_num += 1


def create_source_data(y, class_num, source_train_number):
    train_index = np.array([])
    test_index = np.array([])
    for i in range(0, class_num + 1):
        index = np.where(y == i)[0]
        random.shuffle(index)
        train_number = source_train_number
        train = index[0:train_number]
        test = index[train_number:]
        train_index = np.hstack([train_index, train])
        test_index = np.hstack([test_index, test])
    return train_index, test_index


def create_target_data(y, class_num, target_train_number):
    train_index = np.array([])
    test_index = np.array([])
    for i in range(0, class_num + 1):
        index = np.where(y == i)[0]
        random.shuffle(index)
        train_number = target_train_number
        train = index[0:train_number]
        test = index[train_number:]
        train_index = np.hstack([train_index, train])
        test_index = np.hstack([test_index, test])
    return train_index, test_index


for times in range(1, 11):
    a, e = create_source_data(y_source, class_num, source_train_number)
    b, d = create_target_data(y_target, class_num, target_train_number)

    np.save("./data1/model_source" + str(times) + ".npy", a.astype("int64"))
    np.save("./data1/model_target" + str(times) + ".npy", b.astype("int64"))
