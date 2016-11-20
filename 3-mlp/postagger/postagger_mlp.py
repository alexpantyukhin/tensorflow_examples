#!/usr/bin/env python
#coding:utf8
import tensorflow as tf
import numpy as np
import os
from dataset import Dataset
from time import time
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs',100,'epochs')
flags.DEFINE_integer('batch_size',128,'batch_size')
flags.DEFINE_integer('word_dim',100,'word dim')
flags.DEFINE_integer('emb_size',23769,'embedding size')
flags.DEFINE_integer('input_units',300,'input layer units')
flags.DEFINE_integer('hidden1_units',128,'hidden1 layer units')
flags.DEFINE_integer('hidden2_units',64,'hidden2 layer units')
flags.DEFINE_integer('classes',45,'the output classes')
flags.DEFINE_float('dropout',0.5,'dropout')
flags.DEFINE_float('learning_rate',1e-3,'learning rate')
flags.DEFINE_string('data_path','../../data/PTB','the dataset path')

def mlp():
    
    train_data = Dataset(os.path.join(FLAGS.data_path,'penn.train.pos'))
    dev_data = Dataset(os.path.join(FLAGS.data_path,'penn.devel.pos'))

    embedding = tf.get_variable("embedding",[FLAGS.emb_size,FLAGS.word_dim],tf.float32)
    
    words_hash = tf.placeholder(tf.int32,[None,3])
    y_ = tf.placeholder(tf.int32,[None])

    x = tf.reshape(tf.nn.embedding_lookup(embedding,words_hash),(-1,FLAGS.input_units))

    #first layer
    with tf.variable_scope('hidden1'):
        weights = tf.get_variable('weight',[FLAGS.input_units,FLAGS.hidden1_units])
        biases = tf.get_variable('biases',[FLAGS.hidden1_units])
        hidden1 = tf.nn.relu(tf.matmul(x,weights) + biases)
    #second layer
    with tf.variable_scope('hidden2'):
        weights = tf.get_variable('weight',[FLAGS.hidden1_units,FLAGS.hidden2_units])
        biases = tf.get_variable('weights',[FLAGS.hidden2_units])
        hidden2 = tf.nn.relu(tf.matmul(hidden1,weights) + biases)
    #dropout layer
    keep_prob = tf.placeholder(tf.float32) 
    hidden2_drop = tf.nn.dropout(hidden2,keep_prob)
    #linear
    with tf.variable_scope('linear'):
        weights = tf.get_variable('weights',[FLAGS.hidden2_units,FLAGS.classes])
        biases = tf.get_variable('biases',[FLAGS.classes])
        logits = tf.matmul(hidden2_drop,weights) + biases
    #loss
    with tf.variable_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,y_,name = 'cross_entropy')
        loss = tf.reduce_mean(cross_entropy,name = 'xentropy')
    #train
    train = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
    #eval
    correct_prediction = tf.nn.in_top_k(logits,y_,1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        start = time()
        sess.run(tf.initialize_all_variables())
        cur_epoch = 0
        dev_words,dev_labels = dev_data.all_data()
        while train_data.epochs() < FLAGS.epochs:
            if cur_epoch < train_data.epochs():
                cur_epoch += 1
                print "epoch: "+str(cur_epoch)
                print accuracy.eval(feed_dict={words_hash:dev_words,y_:dev_labels,keep_prob:1})
            batch_x,batch_y = train_data.next_batch(FLAGS.batch_size)
            sess.run(train,feed_dict={words_hash:batch_x,y_:batch_y,keep_prob:1-FLAGS.dropout})
        print "last epoch:"
        print accuracy.eval(feed_dict={words_hash:dev_words,y_:dev_labels,keep_prob:1})
        end = time()
        print 'time:' + str(end-start) + ' seconds'
def main(_):
    mlp()
if __name__=='__main__':
    tf.app.run()
