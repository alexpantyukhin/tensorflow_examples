#!/usr/bin/env python
#coding:utf8
from __future__ import division

import tensorflow as tf
from input_data import read_dataset
from copy import deepcopy
import numpy as np
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
flags.DEFINE_integer('is_training',1,'is training')

def evalution(sess,correct,x_pl,y_pl,mask_pl,output_keep_prob_pl,seq_len_pl,dataset):
    n_epoch = dataset.numbers() // FLAGS.batch_size
    tokens = 0
    corrects = 0
    for step in range(n_epoch):
        batch_x,batch_y,mask_seed = util.padding(*dataset.next_batch(FLAGS.batch_size))
        sequence_length = batch_x.shape[1] * np.ones([FLAGS.batch_size],np.int32)
        tokens += mask_seed.sum()
        corrects += sess.run(correct,
                feed_dict={x_pl:batch_x,y_pl:batch_y,output_keep_prob_pl:1,
                    mask_pl:mask_seed, seq_len_pl:sequence_length})
    print corrects,tokens,corrects/tokens

def bi_lstm():
    tf.set_random_seed(1)
    # load data
    train_data = read_dataset(os.path.join(FLAGS.data_path,'penn.train.pos'))
    dev_data = read_dataset(os.path.join(FLAGS.data_path,'penn.devel.pos'))
    with tf.device('/cpu:0'):    
        embedding = tf.get_variable("embedding",[FLAGS.emb_size,FLAGS.word_dim],tf.float32)
    
    with tf.name_scope('placeholder'):
        x_ = tf.placeholder(tf.int32,[FLAGS.batch_size,None])
        y_ = tf.placeholder(tf.int32,[None])
        mask = tf.placeholder(tf.int32,[None])
        output_keep_prob = tf.placeholder(tf.float32)
        seq_len = tf.placeholder(tf.int32,[None]) 
    # x:[batch_size,n_steps,n_input]
    x = tf.nn.embedding_lookup(embedding,x_)
    with tf.device('/gpu:2'):
        # lstm cell
        lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden)
        lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden)

        # dropout
        lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw,output_keep_prob=1-FLAGS.dropout)
        lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw,output_keep_prob=1-FLAGS.dropout)
        
        # Get lstm cell output
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell_fw, lstm_cell_bw, x, sequence_length = seq_len,dtype=tf.float32)
        outputs = tf.concat(2,outputs)
        outputs = tf.reshape(outputs,[-1,2 * FLAGS.n_hidden])

        # define weights and biases of logistic layer
        with tf.variable_scope('linear'):
            weights = tf.get_variable("weight",[2 * FLAGS.n_hidden,FLAGS.n_classes],tf.float32)
            biases = tf.get_variable("biases",[FLAGS.n_classes],tf.float32)
            logits = tf.matmul(outputs, weights) + biases
            
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits,y_) * tf.cast(mask,tf.float32))
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
        
        y = tf.cast(tf.nn.in_top_k(logits,y_,1),tf.int32) * mask
        correct = tf.reduce_sum(y)
        
        with tf.Session(config=util.gpu_config()) as sess:
            sess.run(tf.global_variables_initializer())
            FLAGS.epoch_size = train_data.numbers() // FLAGS.batch_size
            for step in range(FLAGS.epoch_size * FLAGS.epoch_step):
                batch_x,batch_y,mask_feed = util.padding(*train_data.next_batch(FLAGS.batch_size))
                sequence_length = batch_x.shape[1] * np.ones([FLAGS.batch_size],np.int32)
                sess.run(train_op,
                        feed_dict={x_:batch_x, y_:batch_y, output_keep_prob:1-FLAGS.dropout, 
                            mask:mask_feed, seq_len:sequence_length})
                if step % 100 == 0:
                    evalution(sess,correct,x_,y_,mask,output_keep_prob,seq_len,dev_data)
def main(_):
    bi_lstm()

if __name__=='__main__':
    tf.app.run()
