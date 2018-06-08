# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 18:58:58 2017

@author: NB-SH-001
"""
import tensorflow as tf
import os

tf.reset_default_graph()

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

X = tf.placeholder(tf.float32, shape=[None]) 
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

saver = tf.train.Saver()
#save_file = './Linear_Regression.ckpt'
#feed_dict={X:[10,20,30,40,50]}

sess = tf.Session()
saver.restore(sess, './Linear_Regression.ckpt')
print('W= %3.3f' %sess.run(W),'b= %3.3f' %sess.run(b))

predictions = sess.run(hypothesis, feed_dict= {X:[10,20,30,40,50]})
#print(predictions)
print('\n prediction = ', predictions)











