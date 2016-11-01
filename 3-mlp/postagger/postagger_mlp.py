#!/usr/bin/env python
#coding:utf8
import tensorflow as tf
import numpy as np
from dataset import Dataset

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape))
def bias_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape))
def postagger():
    
    train_data = Dataset('../../data/PTB/penn.train.pos')
    dev_data = Dataset('../../data/PTB/penn.devel.pos')
    embedding = tf.get_variable("embedding",[23769,100],tf.float32)
    
    words_hash = tf.placeholder(tf.int32,[None,3])
    y_ = tf.placeholder(tf.int32,[None])

    x = tf.reshape(tf.nn.embedding_lookup(embedding,words_hash),(-1,300))

    #first layer
    W_fc1 = weight_variable([300,128])
    b_fc1 = bias_variable([128])

    h_1 = tf.nn.relu(tf.matmul(x,W_fc1) + b_fc1)

    #second layer
    W_fc2 = weight_variable([128,64])
    b_fc2 = bias_variable([64])

    h_2 = tf.nn.relu(tf.matmul(h_1,W_fc2) + b_fc2)

    #linear
    W_fc3 = weight_variable([64,45])
    b_fc3 = bias_variable([45])
    
    logits = tf.matmul(h_2,W_fc3) + b_fc3

    #loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,y_)
    loss = tf.reduce_mean(cross_entropy)
    #train
    train = tf.train.AdamOptimizer(0.001).minimize(loss)
    
    #eval
    correct_prediction = tf.nn.in_top_k(logits,y_,1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.initialize_all_variables())
        cur_epoch = 0
        dev_words,dev_labels = dev_data.all_data()
        while train_data.epochs() < 100:
            if cur_epoch < train_data.epochs():
                cur_epoch += 1
                print "epoch: "+str(cur_epoch)
                print accuracy.eval(feed_dict={words_hash:dev_words,y_:dev_labels})
            batch_x,batch_y = train_data.next_batch(128)
            sess.run(train,feed_dict={words_hash:batch_x,y_:batch_y})
        print "last epoch:"
        print accuracy.eval(feed_dict={words_hash:dev_words,y_:dev_labels})
        
if __name__=='__main__':
    postagger()
