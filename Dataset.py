import torch
from PIL import Image
import os
import pickle
import numpy as np


class NOISE_CIFAR10(torch.utils.data.Dataset):

    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    # val_dataset is from data_batch_5

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=True, noise_rate=0,
                 transform=None, target_transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        # now load the picked numpy arrays
        if self.train:
            file_list = self.train_list
        else:
            file_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in file_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape((-1, 3, 32, 32))
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if self.train:
            if noise_rate > 0:
                n_samples = len(self.targets)
                n_noisy = int(noise_rate * n_samples)
                print("%d Noisy samples" % (n_noisy))
                class_index = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
                class_noisy = int(n_noisy / 10)
                noisy_idx = []
                for d in range(10):
                    noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                    noisy_idx.extend(noisy_class_index)
                    print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
                for i in noisy_idx:
                    self.targets[i] = self.other_class(n_classes=10, current_class=self.targets[i])
                print(len(noisy_idx))
                print("Print noisy label generation statistics:")
                for i in range(10):
                    n_noisy = np.sum(np.array(self.targets) == i)
                    print("Noisy class %s, has %s samples." % (i, n_noisy))

    def other_class(self, n_classes, current_class):
        """
        Returns a list of class indices excluding the class indexed by class_ind
        :param nb_classes: number of classes in the task
        :param class_ind: the class index to be omitted
        :return: one random class that != class_ind
        """
        if current_class < 0 or current_class >= n_classes:
            error_str = "class_ind must be within the range (0, nb_classes - 1)"
            raise ValueError(error_str)

        other_class_list = list(range(n_classes))
        other_class_list.remove(current_class)
        other_class = np.random.choice(other_class_list)
        return other_class

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)