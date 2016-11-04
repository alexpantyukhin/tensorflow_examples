#!/usr/bin/env python
#coding:utf8
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate',1e-4,'Initial learning rate')
flags.DEFINE_integer('max_step',2000,'Number of steps to run trainer')
flags.DEFINE_integer('hidden1',128,'Number of units in hidden layer1')
flags.DEFINE_integer('hidden2',32,'Number of units in hidden layer2')
flags.DEFINE_integer('batch_size',100,'Batch Size')
flags.DEFINE_string('data_dir','../../data/MNIST','dataset dir')

def main(_):
    running_train()
if __name__=='__main__':
    tf.app.run()
