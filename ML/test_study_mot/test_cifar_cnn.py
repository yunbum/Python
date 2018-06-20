import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from layers import conv_layer, max_pool_2x2, full_layer

DATA_PATH = "./data/CIFAR10"
BATCH_SIZE = 50
STEPS = 500000


def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vals]
    return out

def unpickle(file):
    with open(os.path.join(DATA_PATH, file), 'rb') as fo:
        u = pickle._Unpickler(fo)
        u.encoding = 'latin1'
        dict = u.load()
    return dict


def display_cifar(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice()] for i in range(size)])
                    for i in range(size)])
    plt.imshow(im)
    plt.show()

class CifaLoader(object):
    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None

    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d["data"] for d in data])
        n = len(images)
        self.images = images.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1)\
            .astype(flaot) / 255
        self.labels = one_hot(np.hstack([d["labels"] for d in data]), 10)
        return self






    
























