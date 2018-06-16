import tensorflow as tf

sess = tf.InteractiveSession()
queue1 = tf.FIFOQueue(capacity=10, dtypes=[tf.string])

enque_op = queue1.enqueue(["F"])
enque_op.run()

print('q size after excution = ',sess.run(queue1.size()))

print(queue1.size())
