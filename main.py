import tensorflow as tf

import numpy as np
import os
import time

def download_dataset(filename='shakespeare.txt', url='https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'):
    path_to_file = tf.keras.utils.get_file(filename, url)

    # Read, then decode for py2 compat.
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    # length of text is the number of characters in it
    print('Length of text: {} characters'.format(len(text)))

    return text

if __name__ == "__main__":
    download_dataset()