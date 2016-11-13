#!/usr/bin/env python
#coding:utf8
import sys
import numpy
import cPickle as pkl
sys.path.append('../../1-utils/')
from dataset import Dataset
'''
embedding[0] = padding
embedding[1] = UNK
'''

def read_dataset(filename):
    label_hash = pkl.load(open('../../data/PTB/label2id.pkl'))
    word_hash = pkl.load(open('../../data/PTB/word2id.pkl'))
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
            label_list.append(label_hash[label] if label in label_hash else 0)
        sents_list.append(word_list)
        labels_list.append(label_list)
    sents = numpy.array(sents_list)
    labels = numpy.array(labels_list)
    dataset = Dataset(sents,labels)
    return dataset

if __name__=='__main__':
    labels = pkl.load(open('../../data/PTB/label2id.pkl'))
    for key in labels.keys():
        print key
