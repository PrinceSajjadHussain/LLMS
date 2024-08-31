import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Parameters
sequence_length = 40
batch_size = 64
epochs = 10
vocab_size = 0

# Data Preprocessing
def load_data():
    # Load Tiny Shakespeare dataset
    dataset, info = tfds.load('tiny_shakespeare', split='train', with_info=True, as_supervised=False)
    return dataset

def preprocess_data(dataset):
    # Initialize the tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)

    # Collect sample texts for building the tokenizer
    def get_texts(data):
        text = data['text'].numpy().decode('utf-8')  # Access the text key in the dictionary
        return text

    texts = [get_texts(data) for data in dataset.take(1000)]
    tokenizer.fit_on_texts(texts)
    
    global vocab_size
    vocab_size = len(tokenizer.word_index) + 1

    def tokenize_text(text):
        tokens = tokenizer.texts_to_sequences([text])[0]
        tokens = tokens[:sequence_length]  # Truncate to sequence_length
        tokens += [0] * (sequence_length - len(tokens))  # Pad to sequence_length
        return tokens

    def tf_tokenize_text(data):
        text = get_texts(data)  # Extract text
        tokens = tokenize_text(text)  # Tokenize
        return tf.convert_to_tensor(tokens, dtype=tf.int32)

    # Convert the dataset to a list of (text, label) tuples
    tokenized_texts = [tf_tokenize_text(data) for data in dataset]

    # Create a TensorFlow dataset from the list of tuples
    dataset = tf.data.Dataset.from_tensor_slices((tokenized_texts, tf.zeros(len(tokenized_texts))))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, tokenizer

# GAN Model
def build_generator():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(sequence_length, vocab_size)),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dense(vocab_size, activation='softmax')
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(sequence_length,)),
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=sequence_length),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = tf.keras.layers.Input(shape=(sequence_length,))
    x = tf.expand_dims(gan_input, axis=-1)  # Add feature dimension to match generator input
    generated_texts = generator(x, training=False)
    gan_output = discriminator(tf.squeeze(generated_texts, axis=-1))
    model = tf.keras.Model(inputs=gan_input, outputs=gan_output)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# Training GAN
def train_gan():
    dataset, tokenizer = preprocess_data(load_data())
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        for real_texts, _ in dataset:
            noise = tf.random.normal([batch_size, sequence_length])
            generated_texts = generator(tf.expand_dims(noise, axis=-1), training=False)
            real_labels = tf.ones((batch_size, 1))
            fake_labels = tf.zeros((batch_size, 1))
            
            # Train discriminator
            with tf.GradientTape() as tape:
                real_preds = discriminator(tf.squeeze(real_texts, axis=-1), training=True)  # Ensure the right shape
                fake_preds = discriminator(tf.squeeze(generated_texts, axis=-1))
                d_loss_real = tf.keras.losses.binary_crossentropy(real_labels, real_preds)
                d_loss_fake = tf.keras.losses.binary_crossentropy(fake_labels, fake_preds)
                d_loss = d_loss_real + d_loss_fake
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            discriminator.optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
            
            # Train generator
            with tf.GradientTape() as tape:
                generated_texts = generator(tf.expand_dims(noise, axis=-1), training=True)
                fake_preds = discriminator(tf.squeeze(generated_texts, axis=-1))
                g_loss = tf.keras.losses.binary_crossentropy(real_labels, fake_preds)
            grads = tape.gradient(g_loss, generator.trainable_variables)
            gan.optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        
        print(f"Epoch {epoch+1}/{epochs} completed")

if __name__ == "__main__":
    train_gan()












import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# parameters
sequence_length=40
batch_size=64
epochs=10
vocab_size=0

#Data preprocessing
def load_data():
    dataset, info =tfds.load('tiny_shakespeare', split='train', with_info=True, as_supervised=False)
    return dataset

def preprocess_data(dataset):
    #Initialize the tokenizer
    tokenizer= tf.keras.preprocessing.text.Tokenizer(char_level=True)

    #collect sample texts for building the tokenizer
    def get_texts(data):
        text=data['text'].numpy().decode('utf-8')
        return text
    texts = [get_texts(data) for data in dataset.take(1000)]
    tokenizer
