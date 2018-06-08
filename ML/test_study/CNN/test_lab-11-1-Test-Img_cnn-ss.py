# Lab 11 MNIST and Convolutional Neural Network
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
f_path = [
        'C:/Users/NB-SH-001/Documents/Python Scripts/data/img_small/01_bolt/bolt_txt/',
        'C:/Users/NB-SH-001/Documents/Python Scripts/data/img_small/02_gear/gear_txt/',
        'C:/Users/NB-SH-001/Documents/Python Scripts/data/img_small/03_motor/motor_txt/',
#        'C:/Users/NB-SH-001/Documents/Python Scripts/data/img_small/04_screw/screw_txt/',
        'C:/Users/NB-SH-001/Documents/Python Scripts/data/img_small/05_washer/washer_txt/',
#        'C:/Users/NB-SH-001/Documents/Python Scripts/data/img_small/06_warmgear/warmgear_txt/',
        'C:/Users/NB-SH-001/Documents/Python Scripts/data/img_small/07_lego/lego_txt/'
        ]

f_name = [
        '1_28x28_bolt_',
        '1_28x28_gear_',
        '1_28x28_motor_',
#        '1_28x28_screw_',
        '1_28x28_washer_',
#        '1_28x28_warmgear_',
        '1_28x28_lego_'
        ]
#from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility
tf.reset_default_graph()   
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
group_size = 3
db_n = 5 # 7

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, db_n])
keep_prob = tf.placeholder(tf.float32)
# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
'''

# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64])
'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)
'''

# Final FC 7x7x64 inputs -> 10 outputs
W3 = tf.get_variable("W3", shape=[7 * 7 * 64, db_n],
                     initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([db_n]))
logits = tf.matmul(L2_flat, W3) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
#    avg_cost = 0
    print('epoch n =',epoch+1,'#########')
#    total_batch = int(mnist.train.num_examples / batch_size)
#   total_batch = 10
    for db_i in range(db_n): #dataset loop
        for g_i in range(group_size): #array X,Y ini / 
            x_buff3 = np.empty([0,784], dtype = float)
            y_buff1 = np.zeros((batch_size, db_n-1))
            y_buff2 = np.ones(batch_size)
            y_buff3 = np.insert(y_buff1, db_i, y_buff2, 1)
            #batch_x build start
            for b_i in range(batch_size): #each file / 
                f_num = g_i * batch_size + b_i
                file_name = f_path[db_i] + f_name[db_i] + str(f_num) +'.txt'
                x_only = np.loadtxt(file_name, delimiter='	', dtype=np.float32)
                x_buff3 = np.insert(x_buff3, b_i, x_only, axis=0)
#                print('fname ', file_name)
            #batch_x build end /
#            x_buff3 = x_buff3.reshape(-1,100,100,1)
            feed_dict = {X: x_buff3, Y: y_buff3, keep_prob: 0.7}
#            print('y_buff3 = ',y_buff3[0])
            c, _, img = sess.run([cost, optimizer, X_img], feed_dict=feed_dict)
            if g_i == 0:    #check data
                plt.imshow(img[0].reshape(28,28))
                plt.show()
#                print('y_buff =',y_buff3[0])
            print('cost = %3.8f' %c)
#    avg_cost += c / total_batch
#    plt.imshow(img[0].reshape(100,100))
#    plt.show()
#    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

prediction = tf.argmax(logits, 1)
is_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

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
    c, _, img, fred = sess.run([cost, optimizer, X_img, prediction], feed_dict=feed_dict)
    print('prediction', fred)
    print('cost = ', c)
    plt.imshow(img[0].reshape(28,28))
    plt.show()
#
  
    
'''
# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()
'''


'''
Epoch: 0001 cost = 0.340291267
Epoch: 0002 cost = 0.090731326
Epoch: 0003 cost = 0.064477619
Epoch: 0004 cost = 0.050683064
Epoch: 0005 cost = 0.041864835
Epoch: 0006 cost = 0.035760704
Epoch: 0007 cost = 0.030572132
Epoch: 0008 cost = 0.026207981
Epoch: 0009 cost = 0.022622454
Epoch: 0010 cost = 0.019055919
Epoch: 0011 cost = 0.017758641
Epoch: 0012 cost = 0.014156652
Epoch: 0013 cost = 0.012397016
Epoch: 0014 cost = 0.010693789
Epoch: 0015 cost = 0.009469977
Learning Finished!
Accuracy: 0.9885
'''
