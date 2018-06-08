# Lab 11 MNIST and Convolutional Neural Network
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import time

# import matplotlib.pyplot as plt
f_path = [
        '/Users/yummybum/Dropbox/Study/Python_Tensorflow/Datasets/data_img/img_small/01_bolt/bolt_txt/',
        '/Users/yummybum/Dropbox/Study/Python_Tensorflow/Datasets/data_img/img_small/02_gear/gear_txt/',
        '/Users/yummybum/Dropbox/Study/Python_Tensorflow/Datasets/data_img/img_small/03_motor/motor_txt/',
        '/Users/yummybum/Dropbox/Study/Python_Tensorflow/Datasets/data_img/img_small/05_washer/washer_txt/',
        '/Users/yummybum/Dropbox/Study/Python_Tensorflow/Datasets/data_img/img_small/07_lego/lego_txt/'
        ]
#/Users/yummybum/Dropbox/Study/Python_Tensorflow/Datasets/data_img/img_small/01_bolt

f_name = [
        '1_28x28_bolt_',
        '1_28x28_gear_',
        '1_28x28_motor_',
#        '1_28x28_screw_',
        '1_28x28_washer_',
#        '1_28x28_warmgear_',
        '1_28x28_lego_'
        ]

start_T = time.clock()

# initialize
tf.set_random_seed(777)  # reproducibility
tf.reset_default_graph()   # reset previsou W, b

# hyper parameters
learning_rate = 0.001
training_epochs = 50
batch_size = 100
group_size = 3
db_n = 5 # 7

# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, db_n])

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
##
# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])
##
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
#    Conv      ->(?, 7, 7, 128)
#    Pool      ->(?, 4, 4, 128)
#    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])
##
# L4 FC 4x4x128 inputs -> 625 outputs
W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
##
# L5 Final FC 625 inputs -> db_n outputs
W5 = tf.get_variable("W5", shape=[625, db_n],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([db_n]))
logits = tf.matmul(L4, W5) + b5

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, './test_deep_cnn.ckpt')
print('b5', sess.run(b5))

'''
# train my model

'''


prediction = tf.argmax(logits, 1)
'''
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
'''


test_set = 60
for db_i in range(db_n): #dataset loop
    print('\n db file =', f_name[db_i],'######')
    x_buff3 = np.empty([0,784], dtype = float)
    y_buff1 = np.zeros((test_set, db_n-1))
    y_buff2 = np.ones(test_set)
    y_buff3 = np.insert(y_buff1, db_i, y_buff2, 1)
    #batch_x build start
    for f_i in range(test_set): #each file / group_size -> batch_size
        f_num = 300 + f_i
        file_name = f_path[db_i] + f_name[db_i] + str(f_num) +'.txt'
        x_only = np.loadtxt(file_name, delimiter='	', dtype=np.float32)
        x_buff3 = np.insert(x_buff3, f_i, x_only, axis=0)
#        print('f name', file_name)
    #batch_x build end
#    x_buff3 = x_buff3.reshape(-1,100,100,1)
#    print('x_buff3 = ',x_buff3[0])
#    print('y_buff3 = ',y_buff3[0])
    feed_dict = {X: x_buff3, Y: y_buff3, keep_prob: 1.0}
    c, _, img, fred = sess.run([cost, optimizer, X_img, prediction], 
                               feed_dict=feed_dict)
#    y_result = np.full(test_set, db_i)
#    accu = sess.run([accuracy], feed_dict= {Y: y_result})
    plt.imshow(img[0].reshape(28,28))
    plt.show()
    print('prediction', fred)
    print('cost = %3.5f' %c)
end_T = time.clock()
dt = end_T - start_T
print('time lapsed = %3.3f'%dt)

# Test model and check accuracy
'''
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))

'''

