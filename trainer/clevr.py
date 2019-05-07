from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import glob
import json
import re
import tensorflow as tf
from tensorflow.python.lib.io import file_io


# DATASET_ROOT_DIR = '/home/rodrigo/Desktop/CLEVR_v1.0'
DATASET_ROOT_DIR = 'gs://rrdata/CLEVR_v1.0'

IMG_SIZE = 448


def get_splits():

    def preprocess_image(path):
        image_tensor = tf.read_file(path)
        image = tf.image.decode_png(image_tensor, channels=3)
        image = tf.image.resize_images(image, (IMG_SIZE, IMG_SIZE))
        image /= 255.0
        return image

    def preprocess_sentence(sentence):
        q = sentence.lower()[:-1]
        q = re.sub(r"([.,;])", r" \1 ", q)
        q = re.sub(r'\s{2,}', ' ', q)
        # q = q + ' <end>'
        return q

    def parse_question_file(filename):
        with file_io.FileIO(os.path.join(DATASET_ROOT_DIR, 'questions',
                                         filename), 'r') as f:
            d = json.load(f)
        data = d['questions']
        split_dir = os.path.join(DATASET_ROOT_DIR,
                                 'images',
                                 d['info']['split'])
        img_paths = [os.path.join(split_dir, q['image_filename'])
                     for q in data]
        questions = [preprocess_sentence(q['question']) for q in data]
        answers = [q['answer'] for q in data]
        lengths = [len(q) for q in questions]

        return img_paths, questions, answers, lengths

    def convert_text_to_sequences(text, tokenizer):
        tensor = tokenizer.texts_to_sequences(text)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                               padding='post')
        return tensor

    def load_sample(img_path, question, answer, length):
        img_tensor = preprocess_image(img_path)
        return img_tensor, question, answer, length

    def build_split(img_paths, questions, answers, lengths):
        split = tf.data.Dataset.from_tensor_slices(
            (img_paths, questions, answers, lengths)
        )
        split = split.map(load_sample)
        return split

    train_img_paths, train_questions, train_answers, train_lengths = \
        parse_question_file('CLEVR_train_questions.json')
    validation_img_paths, validation_questions, validation_answers, \
        validation_lengths = \
        parse_question_file('CLEVR_val_questions.json')

    questions_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    questions_tokenizer.fit_on_texts(train_questions)
    answers_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    answers_tokenizer.fit_on_texts(train_answers)

    train_questions_tensor = convert_text_to_sequences(train_questions,
                                                       questions_tokenizer)
    train_answers_indexes = [answers_tokenizer.word_index[w] - 1
                             for w in train_answers]

    validation_questions_tensor = convert_text_to_sequences(
        validation_questions,
        questions_tokenizer)
    validation_answers_indexes = [answers_tokenizer.word_index[w] - 1
                                  for w in validation_answers]
    print(len(train_img_paths))
    print(len(train_questions_tensor))
    print(set(train_answers_indexes))
    print(len(validation_img_paths))
    print(len(validation_questions_tensor))
    print(len(validation_answers_indexes))

    train_split = build_split(train_img_paths,
                              train_questions_tensor,
                              train_answers_indexes,
                              train_lengths)

    validation_split = build_split(validation_img_paths,
                                   validation_questions_tensor,
                                   validation_answers_indexes,
                                   validation_lengths)

    return train_split, validation_split, None
