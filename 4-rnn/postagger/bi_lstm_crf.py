#!/usr/bin/env python
#coding:utf8
from __future__ import division

import tensorflow as tf
import numpy as np
import sys
import os
import cPickle as pkl
sys.path.append('../../1-utils')
from input_data import read_dataset
import util

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size',32,'batch_size')
flags.DEFINE_integer('n_hidden',300,'hidden units')
flags.DEFINE_integer('epoch_step',10,'nums of epochs')
flags.DEFINE_integer('n_classes',46,'nums of classes')
flags.DEFINE_integer('emb_size',23769,'embedding size')
flags.DEFINE_integer('word_dim',100,'word dim')
flags.DEFINE_float('learning_rate',1e-3,'learning rate')
flags.DEFINE_float('dropout',0.1,'dropout')
flags.DEFINE_integer('is_training',1,'is training')
flags.DEFINE_float('beta',0.001,'beta of l2 loss')
flags.DEFINE_string('data_path','../../data/PTB','data path')

def evalution(sess,transition_params,dataset,x_,y_,output_keep_prob,test_unary_scores,test_seq_len):
    tokens = 0
    corrects = 0
    for i in range(dataset.numbers() // FLAGS.batch_size):
        batch_x,batch_y,_ = util.padding(*dataset.next_batch(FLAGS.batch_size))
        batch_y = batch_y.reshape([FLAGS.batch_size,-1])
        feed_dict = {x_:batch_x,y_:batch_y,output_keep_prob:1}
        unary_scores , sequence_length = sess.run([test_unary_scores,test_seq_len],feed_dict = feed_dict)
         
        transMatrix = sess.run(transition_params)
        for sent_unary_scores,y,sent_length in zip(unary_scores,batch_y,sequence_length):
            if sent_length != 0:
                sent_unary_scores = sent_unary_scores[:sent_length]
                y = y[:sent_length]
                viterbi_sequence , _ = tf.contrib.crf.viterbi_decode(sent_unary_scores,transMatrix)

                corrects += np.sum(np.equal(viterbi_sequence,y))
                tokens += sent_length
            else:
                continue

    print corrects, tokens, corrects / tokens
    return corrects / tokens 
def bi_lstm_crf():
    # load data
    print 'start read dataset'
    train_data = read_dataset(os.path.join(FLAGS.data_path,'penn.train.pos'))
    dev_data = read_dataset(os.path.join(FLAGS.data_path,'penn.devel.pos'))
    dev_data.fake_data(FLAGS.batch_size)
    print 'stop read dataset'
    
    tf.set_random_seed(1)

    
    # 词向量放到cpu里面可以节省显存
    with tf.device('/cpu:0'):
        with tf.variable_scope('embedding') as scope:
            random_embedding = tf.get_variable(
                    name = "random_embedding",
                    shape = [FLAGS.emb_size,FLAGS.word_dim],
                    dtype = tf.float32)
         
    with tf.name_scope('placeholder'):
        x_ = tf.placeholder(tf.int32,[FLAGS.batch_size,None])
        y_ = tf.placeholder(tf.int32,[FLAGS.batch_size,None])
        output_keep_prob = tf.placeholder(tf.float32)
    
    sequence_length = tf.reduce_sum(tf.sign(x_),reduction_indices = 1)
    sequence_length = tf.cast(sequence_length,tf.int32)
    
    with tf.device('/gpu:2'):
        with tf.variable_scope('input_layer'):
            # x:[batch_size,n_steps,n_input]
            x = tf.nn.embedding_lookup(random_embedding,x_)
        # lstm cell
        with tf.name_scope('bi_lstm_layer'):
            lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden)
            lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.n_hidden)

        # Get lstm cell output
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = lstm_cell_fw, 
                cell_bw = lstm_cell_bw, 
                inputs = x,
                sequence_length = sequence_length,
                dtype=tf.float32)
        outputs = tf.concat(2,outputs)
        outputs = tf.reshape(outputs,[-1,2 * FLAGS.n_hidden])
        
        outputs = tf.nn.dropout(outputs,keep_prob = output_keep_prob)
        with tf.variable_scope('Softmax'):
            weights = tf.get_variable(
                    name = "weights",
                    shape = [2 * FLAGS.n_hidden,FLAGS.n_classes],
                    dtype = tf.float32,
                    initializer = tf.truncated_normal_initializer(stddev=0.01))
            biases = tf.get_variable(
                    name = "biases",
                    shape = [FLAGS.n_classes],
                    dtype = tf.float32)
        matricized_unary_scores = tf.matmul(outputs,weights) + biases
        unary_scores = tf.reshape(matricized_unary_scores,[FLAGS.batch_size, -1 , FLAGS.n_classes])

        log_likelihood , transition_params = tf.contrib.crf.crf_log_likelihood(unary_scores,y_,sequence_length)
        l2_loss = tf.nn.l2_loss(weights) * FLAGS.beta
        loss = tf.reduce_mean(-log_likelihood) + l2_loss
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
        
    saver = tf.train.Saver()
    best_acc = 0
    if FLAGS.is_training == 1:
        with tf.Session(config=util.gpu_config()) as sess:
            sess.run(tf.global_variables_initializer())
            epoch_size = train_data.numbers() // FLAGS.batch_size
            for step in range(epoch_size * FLAGS.epoch_step):
                batch_x,batch_y,_ = util.padding(*train_data.next_batch(FLAGS.batch_size))
                sess.run([l2_loss,loss,train_op],feed_dict={x_:batch_x, y_:batch_y.reshape([FLAGS.batch_size,-1]), output_keep_prob:1-FLAGS.dropout})
                if step % 100 == 0:
                    cur_acc = evalution(sess,transition_params,dev_data,x_,y_,output_keep_prob,unary_scores,sequence_length)
                    if cur_acc > best_acc:
                        best_acc = cur_acc
                        #saver.save(sess,'best.model')
        print 'best_acc: ' + str(best_acc)
    else:
        pass
def main(_):
    bi_lstm_crf()
if __name__=='__main__':
    tf.app.run()
