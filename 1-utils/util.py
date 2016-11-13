#!/usr/bin/env python
#coding:utf8
from copy import deepcopy
import numpy
import tensorflow as tf

def gpu_config():
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    return config

def padding(data,labels):

    assert isinstance(data,numpy.ndarray),'the data type should be numpy.ndarray whose dtype is list'
    assert isinstance(labels,numpy.ndarray),'the label type should be numpy.ndarray whose dtype is list'
    assert data.shape == labels.shape,'data shape and labels shape should be same'

    max_seg = max([len(instance) for instance in data])
    batch_size = data.shape[0]
    
    batch_x = numpy.zeros([batch_size, max_seg],numpy.int32)
    batch_y = numpy.zeros([batch_size, max_seg],numpy.int32)
    mask = numpy.zeros([batch_size, max_seg],numpy.int32)

    for i in range(batch_size):
        cur_len = len(data[i])
        zeros = [0 for _ in range(max_seg - cur_len)]
        ones = [1 for _ in range(cur_len)]
        batch_x_i = deepcopy(data[i])
        batch_y_i = deepcopy(labels[i])
        if cur_len < max_seg:
            batch_x_i.extend(zeros)
            batch_y_i.extend(zeros)
            ones.extend(zeros)
         
        batch_x[i] = batch_x_i
        batch_y[i] = batch_y_i
        mask[i] = ones
    batch_y = batch_y.reshape([batch_size * max_seg])
    mask = mask.reshape([batch_size * max_seg])
    return batch_x,batch_y,mask
