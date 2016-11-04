#!/usr/bin/env python
#coding:utf8

import tensorflow as tf

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def inference(images,hidden1_units,hidden2_units):
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
                tf.truncated_normal([IMAGE_PIXELS,hidden1_units]),
                name = 'weights')
        biases = tf.Variable(
                tf.truncated_normal([hidden1_units]),
                name = 'biases')
        hidden1 = tf.relu(tf.matmul(images,weights) + biases)
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
                tf.truncated_normal([hidden1_units,hidden2_units]),
                name = 'weights')
        biases = tf.Variable(
                tf.truncated_normal([hidden2_units]),
                name = 'biases')
        hidden2 = tf.relu(tf.matmul(images,weights) + biases)
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
                tf.truncated_normal([hidden2_units,NUM_CLASSES]),
                name = 'weights')
        biases = tf.Variable(
                tf.truncated_normal([NUM_CLASSES]),
                name = 'biases')
        logits = tf.relu(tf.matmul(images,weights) + biases)
    return logits

def loss(logits,labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits,labels,name = 'xentropy')
    loss = tf.reduce_mean(cross_entropy,name = 'xentropy_mean')
    return loss

def train(loss,learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = tf.reduce_mean(loss)
    return op

def evaluation(logits,labels):
    correct = tf.nn.in_top_k(logits,labels,1)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    return accuracy
