#!/usr/bin/env python
#coding:utf8
import sys
import numpy
import cPickle as pkl

'''
embedding[0] = padding
embedding[1] = UNK
'''
def read_dataset(filename):
    label_hash = pkl.load(open('../../data/PTB/label_hash.pkl'))
    word_hash = pkl.load(open('../../data/PTB/word_hash.pkl'))
    words_list = []
    labels_list = []
    for line in open(filename):
        tokens = line.strip().split()
        for i in range(len(tokens)):
            word_list = []   
            #add previous word
            if i == 0:
                word_list.append(0)
            else:
                pre_word = tokens[i-1].rsplit('/',1)[0]
                word_list.append(word_hash[pre_word] if pre_word in word_hash else 1)
            #add cur word
            cur_word = tokens[i].rsplit('/',1)[0]
            word_list.append(word_hash[cur_word] if cur_word in word_hash else 1)
            #add next word
            if i == len(tokens) - 1:
                word_list.append(0)
            else:
                next_word = tokens[i+1].rsplit('/',1)[0]
                word_list.append(word_hash[next_word] if next_word in word_hash else 1)
            words_list.append(word_list)
            #add cur label
            label = tokens[i].rsplit('/',1)[1]
            labels_list.append([label_hash[label] if label in label_hash else 0])

    words = numpy.array(words_list,numpy.int32)
    labels = numpy.array(labels_list,numpy.int32).reshape([len(labels_list),])
    return words,labels

class Dataset():
    def __init__(self,filename):
        words,labels = read_dataset(filename)
        self._words = words
        self._labels = labels
        self._numbers = self._words.shape[0]
        self._index_in_epoch = 0
        self._epochs = 0
    def numbers(self):
        return self._numbers
    def epochs(self):
        return self._epochs
    def all_data(self):
        return self._words,self._labels
    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._numbers:
            #finished epoch
            self._epochs += 1
            #shuffle data
            perm = numpy.arange(self._numbers)
            numpy.random.shuffle(perm)
            self._words = self._words[perm]
            self._labels = self._labels[perm]
            #start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._numbers
        end = self._index_in_epoch
        return self._words[start:end],self._labels[start:end]
if __name__=='__main__':
    word_count = pkl.load(open('word_count.pkl'))
    word_hash = {}
    count = 2
    for key,value in word_count.items():
        if value > 1:
            word_hash[key] = count
            count += 1
    pkl.dump(word_hash,open('word_hash.pkl','w'))
    print len(word_hash)
