#!/usr/bin/env python
#coding:utf8
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import model
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate',1e-3,'Initial learning rate')
flags.DEFINE_integer('max_step',20000,'Number of steps to run trainer')
flags.DEFINE_integer('hidden1',128,'Number of units in hidden layer1')
flags.DEFINE_integer('hidden2',32,'Number of units in hidden layer2')
flags.DEFINE_integer('batch_size',100,'Batch Size')
flags.DEFINE_string('data_dir','../../data/MNIST','dataset dir')

def placeholder_inputs():
    images_placeholder = tf.placeholder(tf.float32,shape=(None,model.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32,shape=(None))
    return images_placeholder,labels_placeholder
def fill_feed_dict(data_set,images_pl,labels_pl):
    images_feed,labels_feed = data_set.next_batch(FLAGS.batch_size)
    feed_dict = {images_pl:images_feed,labels_pl:labels_feed}
    return feed_dict
def do_eval(sess,
        eval_correct,
        images_placeholder,
        labels_placeholder,
        data_set):
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples / FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,images_placeholder,labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples
    print precision
def run_training():
    data_set = input_data.read_data_sets(FLAGS.data_dir)
    with tf.Graph().as_default():
        images_placeholder,labels_placeholder = placeholder_inputs()
        logits = model.inference(images_placeholder,FLAGS.hidden1,FLAGS.hidden2)
        loss = model.loss(logits,labels_placeholder)
    
        train_op = model.train(loss,FLAGS.learning_rate)
        accuracy = model.evaluation(logits,labels_placeholder)
        init = tf.initialize_all_variables()
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init)
            for step in range(FLAGS.max_step):
                feed_dict = fill_feed_dict(data_set.train,images_placeholder,labels_placeholder)
                sess.run(train_op,feed_dict=feed_dict)
                if step % 100 == 0:
                    do_eval(sess,accuracy,images_placeholder,labels_placeholder,data_set.test)
                    #feed_dict = fill_feed_dict(data_set.test,images_placeholder,labels_placeholder,data_set.test.num_examples)
                    #print accuracy.eval(feed_dict=feed_dict)
def main(_):
    run_training()
if __name__=='__main__':
    tf.app.run()
