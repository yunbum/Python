import os
from glob import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time


#전처리 시작
list_training = glob('/Users/ybbaek/PycharmProjects/Datasets/data_img/img_metal_training/*/*/*.jpg')
list_test = glob('/Users/ybbaek/PycharmProjects/Datasets/data_img/img_metal_test/*/*/*.jpg')

#tool funtion def
def get_label_from_path(path):
    str = (path.split('/')[-3])
    return int(str.split('_')[0])

def read_image(path):
    image = np.array(Image.open(path))
    image = np.reshape(image, np.product(image.shape))
    return image

def onehot_encode_label(path):
    onehot_label = unique_label_names == get_label_from_path(path)
    onehot_label = onehot_label.astype(np.uint8)
    return onehot_label

#Training Data, image check
print(get_label_from_path(list_training[0]))
image = np.array(Image.open(list_training[0]))
print(image.shape)

#Test Data check
print(get_label_from_path(list_test[0]))
image = np.array(Image.open(list_test[0]))
print(image.shape)

'''
for i in range(7):
    plt.subplot(1,7, i+1)
    plt.axis('off')
    plt.imshow(np.array(Image.open(list_training[i*300])))
    plt.show()
'''

#Training image array creation
label_name_list = []
for path in list_training:
    label_name_list.append(get_label_from_path(path))
#Label check
unique_label_names = np.unique(label_name_list)
print(unique_label_names)
print(onehot_encode_label(list_training[0]))

#Test image array creation
for path in list_training:
    label_name_list.append(get_label_from_path(path))
unique_label_names = np.unique(label_name_list)
print(unique_label_names)
print(onehot_encode_label(list_test[0]))

#test pram set
batch_size = 30
test_size = 60
img_h = 28
img_w = 28
ch_n = 1
num_class = 7
num_files = len(label_name_list) #2100

Img1D_size = 784

#training batch buff ini
batch_image = np.zeros((batch_size, Img1D_size))
batch_label = np.zeros((batch_size, num_class))
#test batch buff ini
test_image = np.zeros((test_size, Img1D_size))
test_label = np.zeros((test_size, num_class))

# data check
print('batch image shape = ',batch_image.shape)
print('batch label shape = ',batch_label.shape)
print('test image shape = ', test_image.shape)
print('test label shape = ', test_label.shape)

#plt.title(batch_label[0])
#plt.imshow(np.reshape(batch_image[0],(28,28)))
#plt.show()

#model
import tensorflow as tf
from layers import conv_layer, max_pool_2x2, full_layer

x = tf.placeholder(tf.float32, shape=[None, 784])
y_= tf.placeholder(tf.float32, shape=[None,  7])

x_image = tf.reshape(x, [-1, 28, 28, 1])

conv1 = conv_layer(x_image, shape=[5, 5, 1, 32])
conv1_pool = max_pool_2x2(conv1)
conv2 = conv_layer(conv1_pool, [5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)
conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

#hyper param
keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1, keep_prob = keep_prob)
y_conv = full_layer(full1_drop, 7)

#cost function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Test starting...
STEPS = 9

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 전체 epoch
    for i in range(STEPS):
        # time check
        start_t = time.clock()
        # 배치세트 70 <- 2100/30
        for j in range(num_files // batch_size):
            # 배치단위 생성 30ea
            for n, path in enumerate(list_training[batch_size * j:batch_size * (j + 1)]):
                image = read_image(path)
                onehot_label = onehot_encode_label(path)
                batch_image[n, :] = image
                batch_label[n, :] = onehot_label

        if i % 3 ==0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_image, y_:batch_label, keep_prob: 1.0})
            print("step {:4d}, training accuracy = {:.3f}".format(i, train_accuracy))

        sess.run(train_step, feed_dict={x: batch_image, y_: batch_label, keep_prob: 0.5})

        #dt check
        end_t = time.clock()
        print('learning time =%3.3f' % (end_t - start_t))

    X = test_image
    Y = test_label
    print('X shape = ', X.shape)
    print('Y shape = ', Y.shape)

    start_t = time.clock()
    test_accuracy = sess.run(accuracy, feed_dict={x: X, y_: Y, keep_prob: 1.0})

print("test accuracy = : {:.3f}".format(test_accuracy))
end_t = time.clock()
print('\n test time =%3.3f' % (end_t - start_t))