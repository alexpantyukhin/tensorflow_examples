#!/usr/bin/env python
#coding:utf8
import sys
import numpy
import os
import cPickle as pkl
sys.path.append('../../1-utils')
from dataset import Dataset
def read_dataset(data):
    if os.path.isfile(data+'.pkl'):
        return pkl.load(open(data+'.pkl'))
    label2id = pkl.load(open('../../data/PTB/label2id.pkl'))
    word2id = pkl.load(open('../../data/PTB/word2id.pkl'))
    sents_list = []
    labels_list = []
    for line in open(data):
        tokens = line.strip().split()
        word_list = []
        label_list = []
        for i in range(len(tokens)):
            word = tokens[i].rsplit('/',1)[0]
            word_list.append(word2id[word] if word in word2id.keys() else 1)
            label = tokens[i].rsplit('/',1)[1]
            assert label in label2id.keys(),'unknow label'
            label_list.append(label2id[label])
        
        sents_list.append(word_list)
        labels_list.append(label_list)
    dataset = Dataset(sents_list,labels_list)
    pkl.dump(dataset,open(data + '.pkl','w'))
    return dataset
