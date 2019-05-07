from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import tensorflow as tf
import trainer.clevr as clevr
import trainer.model as model

BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
NUM_CLASSES = 28
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
MOMENTUM = 0.9
# SUMMARIES_DIR = '/home/rodrigo/PycharmProjects/baseline/summaries'
SUMMARIES_DIR = 'gs://rrdata/summaries2'

train, validation, test = clevr.get_splits()
train = train.batch(BATCH_SIZE).shuffle(60).prefetch(10)
validation = validation.batch(EVAL_BATCH_SIZE).prefetch(10)

iterator = tf.data.Iterator.from_structure(train.output_types,
                                           train.output_shapes)
train_init_op = iterator.make_initializer(train)
validation_init_op = iterator.make_initializer(validation)

rgb, questions, answers, lengths = iterator.get_next()

predictions = model.model(rgb, questions, lengths, NUM_CLASSES)
predicted_labels = tf.argmax(predictions, 1)

loss = tf.losses.sparse_softmax_cross_entropy(labels=answers,
                                              logits=predictions)

optimizer = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE,
                                       momentum=MOMENTUM)
train_op = optimizer.minimize(loss)

with tf.variable_scope('metrics'):
    acc, acc_op = tf.metrics.accuracy(labels=answers,
                                      predictions=predicted_labels)

validation_accuracy_summary = tf.summary.scalar('validation_accuracy',
                                                acc)

stream_vars = [i for i in tf.local_variables()
               if i.name.split('/')[0] == 'metrics']
reset_op = [tf.initialize_variables(stream_vars)]

init = tf.global_variables_initializer()

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    validation_writer = tf.summary.FileWriter(SUMMARIES_DIR, sess.graph)
    init.run()
    sess.run(reset_op)
    for epoch in range(NUM_EPOCHS):
        print(epoch, end=': ')
        sess.run(train_init_op)
        while True:
            try:
                sess.run(train_op)
            except tf.errors.OutOfRangeError:
                break
        sess.run(validation_init_op)
        while True:
            try:
                accuracy = sess.run(acc_op)
            except tf.errors.OutOfRangeError:
                break
        print(accuracy)
        summary = sess.run(merged)
        validation_writer.add_summary(summary, epoch)
        sess.run(reset_op)
