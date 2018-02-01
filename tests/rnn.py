import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

import tensorflow as tf
import tensorlayer as tl

from tensorflow.contrib.rnn import BasicLSTMCell

input_seqs = tf.placeholder(dtype=tf.int32, shape=[32, None], name="input_seqs")
label = tf.placeholder(dtype=tf.int32, shape=[32], name="label")

net = tl.layers.EmbeddingInputlayer(
    inputs=input_seqs,
    vocabulary_size=500,
    embedding_size=100,
    name='seq_embedding')

net = tl.layers.BiDynamicRNNLayer(net,
                                  cell_fn=BasicLSTMCell,
                                  n_hidden=100,
                                  dropout=0.7,
                                  n_layer=3,
                                  sequence_length=tl.layers.retrieve_seq_length_op2(input_seqs),
                                  name='dynamic_rnn')

net = tl.layers.LambdaLayer(net, fn=lambda x: tf.reduce_sum(x, axis=1))

net = tl.layers.DenseLayer(net, n_units=10, act=tf.identity, name="output")

cost = tl.cost.cross_entropy(net.outputs, label, 'cost')
train_op = tf.train.AdamOptimizer().minimize(cost, var_list=net.all_params)

import numpy as np

X = np.int32(np.random.uniform(low=0, high=400, size=[500, 50]))
Y = np.int32(np.random.uniform(low=0, high=10, size=[500]))

sess = tf.InteractiveSession()
tl.layers.initialize_global_variables(sess)
tl.utils.fit(sess, net, train_op, cost, X, Y, input_seqs, label, batch_size=32, print_freq=5)
