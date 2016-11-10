#!/usr/bin/env python
#coding:utf8
from __future__ import division

import tensorflow as tf
from input_data import read_dataset
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate',1e-3,'learning rate')
flags.DEFINE_integer('batch_size',1,'batch_size')
flags.DEFINE_integer('n_input',100,'the input dim at each time t')
flags.DEFINE_integer('n_hidden',100,'hidden units')
flags.DEFINE_integer('epoch_step',10,'nums of epochs')
flags.DEFINE_integer('epoch_size',1000,'batchs of each epoch')
flags.DEFINE_integer('n_classes',45,'nums of classes')

def evalution(sess,words_hash,y,correct_num,data_set):
    n_tokens = 0
    n_correct = 0
    n_instance = data_set.numbers()
    for step in range(n_instance):
        batch = data_set.next_batch(1)
        batch_x = np.reshape(np.array(batch[0][0]),[FLAGS.batch_size,-1])
        batch_y = np.reshape(np.array(batch[1][0]),[-1])
        n_tokens += batch_y.shape[0]
        n_correct += sess.run(correct_num,feed_dict={words_hash:batch_x,y:batch_y})
    print n_correct / n_tokens
def main(_):
    train_data = read_dataset('../../data/PTB/penn.train.pos')
    dev_data = read_dataset('../../data/PTB/penn.devel.pos')
    
    embedding = tf.get_variable("embedding",[23769,100],tf.float32)
    words_hash = tf.placeholder(tf.int32,[FLAGS.batch_size,None])
    y = tf.placeholder(tf.int32,[None])
    # x:[batch_size,n_steps,n_input]
    x = tf.nn.embedding_lookup(embedding,words_hash)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden,state_is_tuple=True,activation=tf.nn.relu)

    # Get lstm cell output
    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    outputs = tf.reshape(outputs,[-1,FLAGS.n_hidden])
    # define weights and biases of logistic layer
    weights = tf.get_variable("weight",[FLAGS.n_hidden,FLAGS.n_classes])
    biases = tf.get_variable("biases",[FLAGS.n_classes])
    # Linear activation, using rnn inner loop last output
    logits = tf.matmul(outputs, weights) + biases
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits,y))
    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
    
    prediction = tf.nn.in_top_k(logits,y,1)
    correct_num = tf.reduce_sum(tf.cast(prediction,tf.int32))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.initialize_all_variables())
        FLAGS.epoch_size = train_data.numbers()
        step = 0
        while step < FLAGS.epoch_size * FLAGS.epoch_step:    
            batch = train_data.next_batch(1)
            batch_x = np.reshape(np.array(batch[0][0]),[FLAGS.batch_size,-1])
            batch_y = np.reshape(np.array(batch[1][0]),[-1])
            sess.run(train_op,feed_dict={words_hash:batch_x,y:batch_y})
            if step % 10000 == 0:
                evalution(sess,words_hash,y,correct_num,dev_data)
            step += 1
if __name__=='__main__':
    tf.app.run()
