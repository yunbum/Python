import os
import gzip
from os import getcwd
import numpy as np
import tensorflow as tf
from six.moves import urllib

class DataFilename(object):

    def __init__(self):
        self.trainingimages_filename = 'train-image-idx3-ubyte.gz'
        self.traininglabels_filename = 'train-labels-idx1-ubyte.gz'
        self.testimages_filename = 't10k-images-idx3-ubyte.gz'
        self.testlabels_filename = 't10k-labels-idx1-ubyte.gz'


class MnistLoader(object):

    def __init__(self):
        self.SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
        self.WORK_DIRECTORY = getcwd() + '/data/mnist'
        self.IMAGE_SIZE = 28
        self.NUM_CHANNELS = 1
        self.PIXEL_DEPTH = 255
        self.NUM_LABELS = 10



    def download_mnist_dataset(self, filename):
        if not tf.gfile.Exists(self.WORK_DIRECTORY):
            tf.gfile.MakeDirs(self.WORK_DIRECTORY)
            print(" %s is not exit" % self.WORK_DIRECTORY)

        filepath = os.path.join(self.WORK_DIRECTORY, filename)

        print('filepath = %s' % filepath)

        if not tf.gfile.Exists(filepath):
            filepath, _ = urllib.request.urlretrieve(self.SOURCE_URL + filename, filepath)
            with tf.gfile.GFile(filepath) as f:
                size = f.size()
                print ('Successfully downloaded', filename, size, 'bytes.')

            print('[download_mnist_dataset] filepath = %s' % filepath)

        return filepath

'''
    def extract_data(self, filename, num_images):
        print('[extract_data] Extracting gzipped data from %s' % filename)

        with gzip.open(filename) as bytestream:
            bytestream.read(16)

            buf     = bytestream.read(self.IMAGE_SIZE * self.IMAGE_SIZE * num_images * self.NUM_CHANNELS)

            data    = np.frombuffer(buffer = buf,
                                    dtype = np.uint8).astype(np.float32)

            data    = (data - (self.PIXEL_DEPTH / 2.0)) / self.PIXEL_DEPTH

            data    = data.reshape(num_images, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS)
            return data


    def extract_label(self, filename, num_images):
        print('[extract_label] Extracting gzipped data from %s' % filename)

        with gzip.open(filename=filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1* num_images)
            label = np.frombuffer(buffer= buf,
                                  dtype=np.uint8).astype(np.int64)
        print('[extract_label] Extracting gzipped data from %s' % filename)

        with gzip.open(filename=filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = np.frombuffer(buffer=buf,
                                   dtype=np.uint8).astype(np.int64)
            print('[extract_label] label = %s' % labels)
            return  labels



'''









