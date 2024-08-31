import tensorflow as tf
import tensorflow_datasets as tfds

def load_data():
    dataset, info = tfds.load('tiny_shakespeare', split='train', with_info=True, as_supervised=False)
    return dataset

def preprocess_data(dataset):
    # Initialize the tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)

    # Collect sample texts for building the tokenizer
    def get_texts(data):
        text = data['text'].numpy().decode('utf-8')  # Access the text key in the dictionary
        return text

    texts = [get_texts(text) for text in dataset.take(1000)]
    tokenizer.fit_on_texts(texts)
    
    vocab_size = len(tokenizer.word_index) + 1
    sequence_length = 40  # Define sequence length

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
    dataset = dataset.batch(batch_size=64)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, tokenizer, vocab_size
