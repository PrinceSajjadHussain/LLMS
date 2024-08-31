from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
import tensorflow as tf

def build_generator(sequence_length, vocab_size):
    model = tf.keras.Sequential()
    model.add(Input(shape=(sequence_length, vocab_size)))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dense(vocab_size, activation='softmax'))
    return model

def build_discriminator(sequence_length, vocab_size):
    model = tf.keras.Sequential()
    model.add(Input(shape=(sequence_length, vocab_size)))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_gan(generator, discriminator, sequence_length, vocab_size):
    # Define the GAN input shape
    gan_input = Input(shape=(sequence_length, vocab_size))

    # Use the generator to create fake text
    generated_text = generator(gan_input)

    # Pass the generated text to the discriminator
    gan_output = discriminator(generated_text)

    # Build the GAN model
    gan = Model(inputs=gan_input, outputs=gan_output)

    # Compile the GAN model
    gan.compile(loss='binary_crossentropy', optimizer='adam')

    return gan
