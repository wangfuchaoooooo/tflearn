"""
This example introduces the use of TFLearn variables to easily implement
Tensorflow variables with custom initialization and regularization.
Note: If you are using TFLearn layers, inititalization and regularization
are directly defined at the layer definition level and applied to inner
variables.
"""

import tensorflow as tf
import tflearn
import tflearn.variables as va

# Loading MNIST dataset
import tflearn.datasets.mnist as mnist

train_X, train_Y, test_X, test_Y = mnist.load_data('data/', one_hot=True)

with tf.Graph().as_default():
    X = tf.placeholder('float', shape=[None, 784])
    Y = tf.placeholder('float', shape=[None, 10])

    def dnn(x):
        with tf.variable_scope('Layer1'):
            W1 = va.variable(name='W1', shape=[784, 256],
                             initializer='uniform_scaling',
                             regularizer='L2')
            b1 = va.variable(name='b1', shape=[256])
            x = tf.nn.tanh(tf.add(tf.matmul(x, W1), b1))

        with tf.variable_scope('Layer2'):
            W2 = va.variable(name='W2', shape=[256, 256],
                             initializer='uniform_scaling',
                             regularizer='L2')
            b2 = va.variable(name='b2', shape=[256])
            x = tf.nn.tanh(tf.add(tf.matmul(x, W2), b2))

        with tf.variable_scope('Layer3'):
            W3 = va.variable(name='W3', shape=[256, 10],
                             initializer='uniform_scaling')
            b3 = va.variable(name='b3', shape=[10])
            x = tf.add(tf.matmul(x, W3), b3)

        return x
    net = dnn(X)

    with tf.name_scope('Summarizes'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1)), tf.float32
        ), name='acc')

    trainOp = tflearn.TrainOp(loss=loss, optimizer=optimizer,
                              metric=accuracy, batch_size=128)

    trainer = tflearn.Trainer(train_ops=trainOp,
                              tensorboard_dir='./tmp/tflearn_logs/',
                              tensorboard_verbose=3)

    trainer.fit({X: train_X, Y: train_Y}, val_feed_dicts={X: test_X, Y: test_Y},
                n_epoch=10, show_metric=True, run_id='Variables_example')
