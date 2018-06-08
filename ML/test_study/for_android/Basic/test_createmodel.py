# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 10:13:54 2018

@author: NB-SH-001
"""
# Source code is based on golbin's tutorial repo.
# Link: https://github.com/golbin/TensorFlow-Tutorials/blob/master/01%20-%20TensorFlow%20Basic/03%20-%20Linear%20Regression.py

# X 와 Y 의 상관관계를 분석하는 기초적인 선형 회귀 모델을 만들고 실행해봅니다.
import tensorflow as tf

tf.reset_default_graph()

x_data = [1,2,3]
y_data = [3,4,5]

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")
#X = tf.placeholder_with_default(tf.float32, name="X")
#Y = tf.placeholder_with_default(tf.float32, name="Y")

#tf.add_to_collection('X', X)

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')
b = tf.Variable(tf.random_uniform([1], 0.0, 2.0), name='b')

#hypothesis = X * W + b
#w4 = tf.multiply(X, W, name="op_to_restore")
#hypothesis = tf.add(b, tf.multiply(X,W), name="op_restore")
hypothesis = tf.add(b, tf.multiply(X,W), name="op_restore")

saver = tf.train.Saver()

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

train_op = optimizer.minimize(cost)

with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   tf.train.write_graph(sess.graph_def, '.', 'tfandroid.pbtxt')

   for step in range(100):
       _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})
       print((step, cost_val, sess.run(W), sess.run(b)))

   saver.save(sess, './tfandroid.ckpt')

   print("\n == Test ==")
   print("X: 5, Y: ", sess.run(hypothesis, feed_dict={X:5}))
   print("X: 2.5, Y: ", sess.run(hypothesis, feed_dict={X:2.5}))
    
