#!/usr/bin/env python
#coding:utf8
class Dataset():
    def __init__(self,data,label):
        self._data = data
        self._labels = label
        self._numbers = self._data.shape[0]
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
