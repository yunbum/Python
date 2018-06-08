# Lab 11 MNIST and Convolutional Neural Network
import tensorflow as tf
import time
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

tf.set_random_seed(777)  # reproducibility
tf.reset_default_graph()   

# hyper parameters
learning_rate = 0.001
training_epochs = 50
batch_size = 100
group_size = 3
db_n = 5 # 7

class Model:
    
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()
        
    def _build_net(self):
        with tf.variable_scope(self.name):
            self.keep_prob = tf.placeholder(tf.float32)
            self.X = tf.placeholder(tf.float32, [None, 784])
            self.X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 5])
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            L1 = tf.nn.conv2d(self.X_img, W1, strides=[1,1,1,1], padding='SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1],
                                strides=[1,2,2,1], padding='SAME')
            L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)
            
            W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
            L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1],
                                strides=[1,2,2,1],padding='SAME')
            L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
            
            W3 = tf.Variable(tf.random_normal([3,3,64,128], stddev=0.01))
            L3 = tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding='SAME')
            L3 = tf.nn.relu(L3)
            L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[
                    1,2,2,1], padding='SAME')
            L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
            L3_flat = tf.reshape(L3, [-1,128*4*4])
            
            W4 = tf.get_variable("W4", shape=[128*4*4,625],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([625]))
            L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
            L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)
            
            W5 = tf.get_variable("W5", shape=[625, 5],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([5]))
            self.logits = tf.matmul(L4, W5) + b5
            
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate).minimize(self.cost)    
        self.prediction = tf.argmax(self.logits, 1)   
        correct_prediction = tf.equal(
                tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
 
    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep_prob: keep_prop})

    def predict2(self, x_test, keep_prop=1.0):
        return self.sess.run(self.prediction, feed_dict={self.X: x_test, self.keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prop: keep_prop})
    
    def train(self, x_data, y_data, keep_prop=0.7):
        return self.sess.run([self.cost, self.optimizer, self.X_img], feed_dict={
                self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})
    
# initialize
sess = tf.Session()
m1 = Model(sess, "m1")
sess.run(tf.global_variables_initializer())

start_t = time.clock()
# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
#    avg_cost = 0
    print('epoch n =',epoch+1,'#########')
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
#            feed_dict = {X: x_buff3, Y: y_buff3, keep_prob: 0.7}
#            print('y_buff3 = ',y_buff3[0])
#            c, _, img = sess.run([cost, optimizer, X_img], feed_dict=feed_dict)

            c, _,img = m1.train(x_buff3, y_buff3)
            if g_i == 0:    #check data
                plt.imshow(img[-1].reshape(28,28))
                plt.imshow(x_only.reshape(28,28))
                plt.show()
                print('y_buff =',y_buff3[0])
            print('cost = %3.4f' %c)
#    avg_cost += c / total_batch
#    plt.imshow(img[0].reshape(100,100))
#    plt.show()
#    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

end_t = time.clock()
print('\n learning time =%3.3f'%(end_t-start_t))
#prediction = tf.argmax(logits, 1)
# is_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
#correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
#    feed_dict = {X: x_buff3, Y: y_buff3, keep_prob: 1.0}
    c, _, img = m1.train(x_buff3, y_buff3)
    print('cost = %3.4f' %c)
    
    fred = m1.predict2(x_buff3)
#    c, _, img, fred = sess.run([cost, optimizer, X_img, prediction], 
#                               feed_dict=feed_dict)
#    y_result = np.full(test_set, db_i)
#    accu = sess.run([accuracy], feed_dict= {Y: y_result})

    plt.imshow(img[0].reshape(28,28))
    plt.show()
    print('prediction', fred)
#    print('accu = ', accu)
#
# Test model and check accuracy
'''
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))

'''



