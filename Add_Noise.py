import pickle
import numpy as np
import os
def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def add_symmetric_noise(eta, root):
    count = 0

    batch_list = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']

    for data_batch in batch_list:
        data_path = os.path.join(root, data_batch)
        data = unpickle(data_path)
        for i in range(10000):
            if np.random.random()< eta:
                data['labels'][i] = np.random.randint(0,10)
                count += 1
        save_path = data_path + '_noise'
        with open(save_path,'wb') as file:
            pickle.dump(data,file)

    print(' total count of noisy label = {}'.format(count))


if __name__ == '__main__':
    root = ''
    eta = 0.4

    add_symmetric_noise(eta, root)

