#!/usr/bin/env python
#coding:utf8

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

def mnist_softmax():
    #load mnist data
    mnist = input_data.read_data_sets('../data/MNIST_data',one_hot=True)

    #build maps
    x = tf.placeholder(tf.float32,shape=[None,784])
    y_ = tf.placeholder(tf.float32,shape=[None,10])

    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    
    y = tf.nn.softmax(tf.matmul(x,W) + b)
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #initialize all variables
        sess.run(tf.initialize_all_variables())
        
        for step in range(100000):
            batch = mnist.train.next_batch(50)
            train_step.run(feed_dict={x:batch[0],y_:batch[1]})
            if step % 100 == 0:
                print accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels})

if __name__=='__main__':
    mnist_softmax()
