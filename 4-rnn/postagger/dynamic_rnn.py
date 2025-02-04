#!/usr/bin/env python
#coding:utf8
from __future__ import division

import tensorflow as tf
from input_data import read_dataset
from copy import deepcopy
import numpy
import os
import sys
sys.path.append('../../1-utils')
import util

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size',50,'batch_size')
flags.DEFINE_integer('n_hidden',300,'hidden units')
flags.DEFINE_integer('epoch_step',100,'nums of epochs')
flags.DEFINE_integer('epoch_size',1000,'batchs of each epoch')
flags.DEFINE_integer('n_classes',46,'nums of classes')
flags.DEFINE_integer('emb_size',23769,'embedding size')
flags.DEFINE_integer('word_dim',100,'word dim')
flags.DEFINE_float('learning_rate',1e-3,'learning rate')
flags.DEFINE_float('dropout',0.5,'dropout')
flags.DEFINE_string('data_path','../../data/PTB','data path')

def evalution(sess,correct,x_pl,y_pl,mask_pl,output_keep_prob_pl,dataset):
    n_epoch = dataset.numbers() // FLAGS.batch_size
    tokens = 0
    corrects = 0
    for step in range(n_epoch):
        batch_x,batch_y,mask_seed = util.padding(*dataset.next_batch(FLAGS.batch_size))
        tokens += mask_seed.sum()
        corrects += sess.run(correct,feed_dict={x_pl:batch_x,y_pl:batch_y,output_keep_prob_pl:1,mask_pl:mask_seed})
    print corrects,tokens,corrects/tokens

def dynamic_rnn():
    # load data
    train_data = read_dataset(os.path.join(FLAGS.data_path,'penn.train.pos'))
    dev_data = read_dataset(os.path.join(FLAGS.data_path,'penn.devel.pos'))
    
    embedding = tf.get_variable("embedding",[FLAGS.emb_size,FLAGS.word_dim],tf.float32)
    
    with tf.name_scope('placeholder'):
        x_ = tf.placeholder(tf.int32,[FLAGS.batch_size,None])
        y_ = tf.placeholder(tf.int32,[None])
        mask = tf.placeholder(tf.int32,[None])
        output_keep_prob = tf.placeholder(tf.float32)
    
    # x:[batch_size,n_steps,n_input]
    x = tf.nn.embedding_lookup(embedding,x_)

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden,state_is_tuple=True,activation=tf.nn.relu)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=1-FLAGS.dropout)
    
    # Get lstm cell output
    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    outputs = tf.reshape(outputs,[-1,FLAGS.n_hidden])
    
    # define weights and biases of logistic layer
    with tf.variable_scope('linear'):
        weights = tf.get_variable("weight",[FLAGS.n_hidden,FLAGS.n_classes],tf.float32)
        biases = tf.get_variable("biases",[FLAGS.n_classes],tf.float32)
        logits = tf.matmul(outputs, weights) + biases
    
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits,y_) * tf.cast(mask,tf.float32))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits,y_))
    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
    
    y = tf.cast(tf.nn.in_top_k(logits,y_,1),tf.int32) * mask
    correct = tf.reduce_sum(y)

    with tf.Session(config=util.gpu_config()) as sess:
        sess.run(tf.initialize_all_variables())
        FLAGS.epoch_size = train_data.numbers() // FLAGS.batch_size
        for step in range(FLAGS.epoch_size * FLAGS.epoch_step):
            batch_x,batch_y,mask_feed = util.padding(*train_data.next_batch(FLAGS.batch_size))
            sess.run(train_op,feed_dict={x_:batch_x, y_:batch_y, output_keep_prob:1-FLAGS.dropout, mask:mask_feed})
            if step % FLAGS.epoch_size == 0:
                evalution(sess,correct,x_,y_,mask,output_keep_prob,dev_data)
def main(_):
    dynamic_rnn()

if __name__=='__main__':
    tf.app.run()
