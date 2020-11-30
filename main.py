import tensorflow as tf

import numpy as np
import os
import time

def download_dataset(filename='shakespeare.txt', url='https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'):
    path_to_file = tf.keras.utils.get_file(filename, url)

    # Read, then decode for py2 compat.
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

    return text

def vectorize(vocab):
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    return char2idx, idx2char

if __name__ == "__main__":
    exploration = True

    text = download_dataset()
    vocab = sorted(set(text))

    char2idx, idx2char = vectorize(vocab)
    text_as_int = np.array([char2idx[c] for c in text])

    if exploration:
        # length of text is the number of characters in it
        print('Length of text: {} characters'.format(len(text)))

        # Take a look at the first 250 characters in text
        print(text[:250])

        # The unique characters in the file
        print('{} unique characters'.format(len(vocab)))

        print('{')
        for char,_ in zip(char2idx, range(20)):
            print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
        print('  ...\n}')

        # Show how the first 13 characters from the text are mapped to integers
        print('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

    # The maximum length sentence you want for a single input in characters
    seq_length = 100
    examples_per_epoch = len(text)//(seq_length+1)

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    if exploration:
        for i in char_dataset.take(5):
            print(idx2char[i.numpy()])

        for item in sequences.take(5):
            print(repr(''.join(idx2char[item.numpy()])))
