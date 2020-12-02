import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Embedding, GRU

def download_dataset(filename='shakespeare.txt', url='https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'):
    path_to_file = tf.keras.utils.get_file(filename, url)

    # Read, then decode for py2 compat.
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

    return text

def vectorize(vocab):
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    return char2idx, idx2char

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]

    return input_text, target_text

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    inputs = Input(batch_shape=(batch_size, 100))
    x = Embedding(vocab_size, embedding_dim)(inputs)
    x = GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform')(x)
    outputs = Dense(vocab_size)(x)
    model = Model(inputs, outputs)

    return model

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def generate_text(model, start_string, num_generate = 1000, temperature = 1.0):
    """Evaluation step (generating text using the learned model)

        Parameters:
                    model (obj): The model that will generate the text
                    start_string (str): Text used to start generation
                    num_generate (int): Number of characters to generate
                    temperature (int): Low temperature results in more predictable text.
                        Higher temperature results in more surprising text.
                        Experiment to find the best setting.

        Returns:
                text (str): returns generated text
    """

    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)

        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

if __name__ == "__main__":
    exploration = False

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
    dataset = sequences.map(split_input_target)

    if exploration:
        for i in char_dataset.take(5):
            print(idx2char[i.numpy()])

        for item in sequences.take(5):
            print(repr(''.join(idx2char[item.numpy()])))

        for input_example, target_example in  dataset.take(1):
            print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
            print('Target data:', repr(''.join(idx2char[target_example.numpy()])))

        for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
            print("Step {:4d}".format(i))
            print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
            print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

    # Batch size
    BATCH_SIZE = 32
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # Length of the vocabulary in chars
    vocab_size = len(vocab)
    # The embedding dimension
    embedding_dim = 256
    # Number of RNN units
    rnn_units = 1024

    model = build_model(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=BATCH_SIZE)

    if exploration:
        for input_example_batch, target_example_batch in dataset.take(1):
            print(input_example_batch.shape)
            example_batch_predictions = model.predict(input_example_batch)
            print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

        sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
        sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

        print(sampled_indices)

        print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
        print()
        print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))

        example_batch_loss = loss(target_example_batch, example_batch_predictions)
        print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
        print("scalar_loss:      ", example_batch_loss.numpy().mean())

    model.compile(optimizer='adam', loss=loss)
    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    EPOCHS = 10

    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    if exploration:
        print(tf.train.latest_checkpoint(checkpoint_dir))

    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))

    print(generate_text(model, u"MAXIMILUS: Hello World"))
