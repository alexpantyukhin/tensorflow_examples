#!/usr/bin/env python
#coding:utf8
import sys
import numpy
import tensorflow as tf
import cPickle as pkl
sys.path.append('../../1-utils/')
from dataset import Dataset
'''
embedding[0] = padding
embedding[1] = UNK
'''
def read_dataset(filename):
    label_hash = pkl.load(open('../../data/PTB/label_hash.pkl'))
    word_hash = pkl.load(open('../../data/PTB/word_hash.pkl'))
    sents_list = []
    labels_list = []
    for line in open(filename):
        tokens = line.strip().split()
        word_list = []
        label_list = []
        for i in range(len(tokens)):
            word = tokens[i].rsplit('/',1)[0]
            word_list.append(word_hash[word] if word in word_hash else 1)
            label = tokens[i].rsplit('/',1)[1]
            label_list.append(label_hash[label])
        sents_list.append(word_list)
        labels_list.append(label_list)
    sents = numpy.array(sents_list)
    labels = numpy.array(labels_list)
    dataset = Dataset(sents,labels)
    return dataset

if __name__=='__main__':
    embedding = tf.get_variable('embedding',[20,10],tf.float32)
    dataset = read_dataset('../../data/PTB/penn.train.pos')
    batch = dataset.next_batch(1)
    batch_x = batch[0][0]
    print batch_x
