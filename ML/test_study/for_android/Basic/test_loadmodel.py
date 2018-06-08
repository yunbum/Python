# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:41:29 2018

@author: NB-SH-001
"""
import tensorflow as tf

tf.reset_default_graph()

#with tf.get_default_graph() as graph:
with tf.Session() as sess:
    
    saver = tf.train.import_meta_graph('tfandroid.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    sess.run(tf.global_variables_initializer())
    #hypothesis = graph.get_operation_by_name("op_restore")
    #W = graph.get_operation_by_name('W')
    
    W = graph.get_tensor_by_name("W:0")
    b = graph.get_tensor_by_name("b:0")
    X = graph.get_tensor_by_name("X:0")

    #X = tf.get_collection('X')[0]
    #X = tf.get_default_graph().get_operation_by_name("X")
    #X = tf.get_collection("placeholder")[0] 
    feed_dict = {X: [4.0]}
    print('sess.run(W) = ', sess.run(W))
    print('sess.run(b) = ', sess.run(b))
    #print('sess.run(b.output.pop()',sess.run(b.output.pop()))

    
    #feed_dict = {X: [4.0]}
    #print('sess.run(X) = ', sess.run(X))

    hypothesis = graph.get_operation_by_name("op_restore")
    print('hypothesis = ',hypothesis)
    print('X = ', X)

    print(sess.run(hypothesis, feed_dict))
    
    
