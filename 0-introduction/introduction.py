#!/usr/bin/env python
#coding:utf8

import tensorflow as tf
import numpy as np

def demo():

    x_data = np.random.rand(100).astype(np.float32)
    y_data = x_data * 0.3 + 0.1

    W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
    b = tf.Variable(tf.zeros([1]))

    y = W * x_data + b

    loss = tf.reduce_mean(tf.square(y-y_data))
    train = tf.train.AdamOptimizer(0.5).minimize(loss)

    init = tf.initialize_all_variables()
    
    #avoid tensorflow grap all GPU resources
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(init)
        for step in range(1000):
            sess.run(train)
            if step % 100 == 0:
                print sess.run(W),sess.run(b)

def main(_):
    demo()
if __name__=='__main__':
    tf.app.run()
