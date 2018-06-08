# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 18:58:58 2017

@author: NB-SH-001
"""
import tensorflow as tf
#import os

tf.reset_default_graph()

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

X = tf.placeholder(tf.float32, shape=[None]) 
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#print('Save Weights: {}'.format(W.name))
#print('Save Bias: {}'.format(b.name))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, W_val, b_val, _= sess.run([cost, W, b, train],
        feed_dict={X:[1,2,3,4,5], Y:[2.1,3.1,4.1,5.1,6.1]})
    if step % 1000 ==0:
        print('step=%4d'%step, ' cost=%3.3f' %cost_val,' W=%2.2f' %W_val,' b=%2.2f' %b_val)

saver = tf.train.Saver()
save_path = saver.save(sess, "./Linear_Regression.ckpt")
print("Model saved in file: ", save_path)







