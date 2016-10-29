#!/usr/bin/env python
#coding:utf8
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import math

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape,stddev = 1.0/math.sqrt(float(28*28))))
def bias_variable(shape):
    return tf.Variable(tf.zeros(shape=shape))
def mnist_mlp():
    mnist = input_data.read_data_sets('../../data/MNIST_data')
    x = tf.placeholder(tf.float32,[None,784])
    y_ = tf.placeholder(tf.int32,[None])

    #first layer
    W_fc1 = weight_variable([784,128])
    b_fc1 = bias_variable([128])

    h_1 = tf.nn.relu(tf.matmul(x,W_fc1) + b_fc1)

    #second layer
    W_fc2 = weight_variable([128,32])
    b_fc2 = bias_variable([32])

    h_2 = tf.nn.relu(tf.matmul(h_1,W_fc2) + b_fc2)

    #linear

    W_fc3 = weight_variable([32,10])
    b_fc3 = bias_variable([10])

    logits = tf.matmul(h_2,W_fc3) + b_fc3

    #loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,y_)
    loss = tf.reduce_mean(cross_entropy)

    #train
    train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    #eval
    correct_prediction = tf.nn.in_top_k(logits,y_,1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.initialize_all_variables())
        for step in range(100000):
            if step % 100 == 0:
                print accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels})
            batch = mnist.train.next_batch(50)
            train.run(feed_dict={x:batch[0],y_:batch[1]})
if __name__=='__main__':
    mnist_mlp()
