
from __future__ import print_function

import tensorflow as tf

import threading
import time

sess = tf.InteractiveSession()

queue1 = tf.FIFOQueue(capacity=10, dtypes=[tf.string])

enque_op = queue1.enqueue(["F"])

print('Q size before enque_op.run() \t',sess.run(queue1.size()), queue1.size())
enque_op.run()
print('Q size after  enque_op.run() \t',sess.run(queue1.size()), queue1.size())
#print('Q size first sess.run2 \t', queue1.size())

enque_op = queue1.enqueue(["I"])
enque_op.run()
enque_op = queue1.enqueue(["F"])
enque_op.run()
enque_op = queue1.enqueue(["O"])
enque_op.run()

print('\nQ size before sess.run() \t\t',sess.run(queue1.size()), queue1.size())

x = queue1.dequeue()
print('x.eval() ',x.eval(),'size = ',sess.run(queue1.size()))
print('x.eval() ',x.eval(),'size = ',sess.run(queue1.size()))
print('x.eval() ',x.eval(),'size = ',sess.run(queue1.size()))
print('x.eval() ',x.eval(),'size = ',sess.run(queue1.size()))

queue1 = tf.FIFOQueue(capacity=10, dtypes=[tf.string], shapes=[()])


print()
gen_random_normal = tf.random_normal(shape=())
queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32], shapes=())
enque = queue.enqueue(gen_random_normal)


def add():
    for i in range(10):
        sess.run(enque)

threads = [threading.Thread(target=add, args=()) for i in range(10)]
#print('threads \n',threads)

for t in threads:
    t.start()

print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))

#print('queue.size before deQ = ',queue.size())
x = queue.dequeue_many(10)
#print('x.eval = \t',x.eval())
print('q size before deQ = ',sess.run(queue.size()))
x.eval()
#print(x.eval())
print('q size after  deQ = ', sess.run(queue.size()))
x.eval()
#print(x.eval())
print('q size after  deQ = ', sess.run(queue.size()))