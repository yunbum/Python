# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:23:32 2018

@author: NB-SH-001
"""

import tensorflow as tf


x = tf.placeholder(tf.float32, name='test')

with tf.Session() as sess:
    graph = tf.
    sess.run(tf.global_variables_initializer())
    feed_dict= {x: [1.0, 2.0, 3.0]}
    
    print(x.op)
    #print(sess.run(x))