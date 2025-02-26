import numpy as np
import copy
from tqdm import tqdm

def data_augmentation(train_set, label, nums=4):
    new_train_set = copy.deepcopy(train_set)
    new_label = copy.deepcopy(label)
    data_num = len(train_set)
    for _ in tqdm(range(nums)):
        for i in tqdm(range(1, data_num - 1)):
            rand_num = int(7500 * np.random.randn() * 0.1)
            tmp_data = copy.deepcopy(train_set[i])
            tmp_data.reshape(7500)
            if rand_num < 0:
                tmp_data = tmp_data[0 : 7500 + rand_num]
                tmp_data = np.append(train_set[i - 1, rand_num:], tmp_data)
            else:
                tmp_data = tmp_data[rand_num:]
                tmp_data = np.append(tmp_data, train_set[i + 1, 0:rand_num])
            tmp_data = np.expand_dims(tmp_data, axis=0)
            tmp_label = copy.deepcopy(label[i])
            tmp_label = np.expand_dims(tmp_label, axis=0)
            new_train_set = np.append(new_train_set, tmp_data, axis=0)
            new_label = np.append(new_label, tmp_label, axis=0)
    return new_train_set, new_label