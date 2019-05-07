from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import vgg


HIDDEN_SIZE = 500
WORD_EMBEDDING_SIZE = 500
IMAGE_VECTOR_SIZE = WORD_EMBEDDING_SIZE
INTERMEDIATE_SIZE = 400
VOCAB_SIZE = 89
NUM_ATTENTION_LAYERS = 4


def image_features(inputs, scope):
    with tf.variable_scope(scope):
        with slim.arg_scope(vgg.vgg_arg_scope()):
            outputs, end_points = vgg.vgg_16(inputs=inputs,
                                             spatial_squeeze=False,
                                             num_classes=0)
        features = end_points[scope + '/vgg_16/pool5']
        transformed_features = tf.layers.conv2d(features,
                                                IMAGE_VECTOR_SIZE,
                                                [1, 1],
                                                use_bias=True,
                                                padding='same',
                                                activation=tf.nn.tanh)
        batch_features = tf.reshape(transformed_features,
                                    (tf.shape(transformed_features)[0],
                                     -1,
                                     transformed_features.shape[3]))

    return batch_features


def question_model(question, sequence_length, scope):
    with tf.variable_scope(scope):
        embedding_matrix = tf.Variable(tf.random_uniform((VOCAB_SIZE,
                                                          WORD_EMBEDDING_SIZE),
                                                         -1, 1))
        word_embeddings = tf.nn.embedding_lookup(embedding_matrix,
                                                 question)

        gru = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
        output, state = tf.nn.dynamic_rnn(
            gru,
            word_embeddings,
            dtype=tf.float32,
            sequence_length=sequence_length,
        )

    return state


def attention_layer(img_features, query_vector, scope):
    with tf.variable_scope(scope):
        w_img = tf.Variable(tf.random_uniform([IMAGE_VECTOR_SIZE,
                                               INTERMEDIATE_SIZE]),
                            name='w_img', dtype=tf.float32)

        formatted_features = tf.reshape(img_features, [-1, IMAGE_VECTOR_SIZE])
        temp_features = tf.matmul(formatted_features, w_img)
        temp_features = tf.reshape(temp_features, [-1, 196, INTERMEDIATE_SIZE])

        w_query = tf.Variable(tf.random_uniform([WORD_EMBEDDING_SIZE,
                                                 INTERMEDIATE_SIZE]),
                              name='w_query', dtype=tf.float32)
        b_query = tf.Variable(tf.random_uniform([1, INTERMEDIATE_SIZE]))
        temp_query_vector = tf.matmul(query_vector, w_query) + b_query
        temp_query_vector = tf.expand_dims(temp_query_vector, 1)
        h_attention = tf.nn.tanh(temp_features + temp_query_vector)

        w_p = tf.Variable(tf.random_uniform([INTERMEDIATE_SIZE, 1]),
                          name='w_p', dtype=tf.float32)
        b_p = tf.Variable(tf.random_uniform([196, 1]),
                          name='b_p', dtype=tf.float32)

        h_attention = tf.reshape(h_attention, [-1, INTERMEDIATE_SIZE])
        scores = tf.matmul(h_attention, w_p)
        scores = tf.reshape(scores, [-1, 196, 1]) + b_p
        attention_weights = tf.nn.softmax(scores, axis=1)
        new_query_vector = attention_weights * img_features
        new_query_vector = tf.reduce_sum(new_query_vector, axis=1) + \
            query_vector
        return new_query_vector


def model(img, question, sequence_length, num_classes):
    img_features = image_features(img, scope='image_features')
    state = question_model(question, sequence_length, scope='question')
    query = state
    for i in range(NUM_ATTENTION_LAYERS):
        query = attention_layer(img_features, query, 'attention%d' % i)
    logits = tf.layers.dense(query, num_classes)
    predictions = tf.nn.softmax(logits)
    return predictions
