import numpy as np
import tensorflow as tf
import requests
import os
import string
import random
import sys

# Configuration
DATA_URL = 'https://www.gutenberg.org/files/100/100-0.txt'  # Shakespeare's Works
LOCAL_FILE = 'shakespeare.txt'
SEQ_LENGTH = 100  # Length of the input sequence
STEP_SIZE = 3     # Stride for the sliding window
BATCH_SIZE = 128
BUFFER_SIZE = 10000
EMBEDDING_DIM = 256
RNN_UNITS = 1024
EPOCHS = 30      # Adjust based on time availability
# Set to True to run a quick demo with less data/epochs
DEMO_MODE = False 

def download_dataset(url, filepath):
    """Downloads the dataset if it doesn't exist."""
    if not os.path.exists(filepath):
        print(f"Downloading dataset from {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            # Encode/decode to handle potential encoding issues cleanly
            text = response.content.decode('utf-8-sig') 
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            sys.exit(1)
    else:
        print("Dataset already exists.")

def preprocess_text(filepath):
    """Reads, cleans, and tokenizes text."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Lowercase
    text = text.lower()

    # Remove punctuation (as per instructions)
    # We create a translation table that maps every punctuation character to None
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # We might want to remove newlines or handle them. 
    # For now, let's keep newlines as they preserve structure (like in plays/poems).
    # However, sometimes excessive whitespace is annoying. Let's purely remove punctuation.
    
    print(f"Corpus length: {len(text)} characters")
    
    # Unique characters
    vocab = sorted(set(text))
    print(f"Unique characters: {len(vocab)}")
    
    # Mappings
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    
    # Convert text to integers
    text_as_int = np.array([char2idx[c] for c in text])
    
    return text, text_as_int, vocab, char2idx, idx2char

def create_dataset(text_as_int, seq_length, batch_size, buffer_size):
    """Creates a tf.data.Dataset for training."""
    # The task is to predict the next character.
    # So input: text[:-1], target: text[1:]
    # But for stateful-like batches or just simple sliding windows:
    
    # We'll use the .from_tensor_slices approach for efficient pipeline
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    
    # Create sequences: seq_length + 1 (input + target char)
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
    
    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text
    
    dataset = sequences.map(split_input_target)
    
    # Shuffle and batch
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    
    return dataset

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    """Builds the LSTM model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=False,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size) # No softmax here because we use from_logits=True in loss
    ])
    return model

def loss_function(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

class TextGenerator(tf.keras.callbacks.Callback):
    """Callback to generate text at the end of every epoch."""
    def __init__(self, model, char2idx, idx2char, seed_text="shall i compare thee"):
        self.sample_model = model
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.seed_text = seed_text

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n--- Generating text after Epoch {epoch+1} ---")
        generated = generate_text(self.sample_model, self.seed_text, self.char2idx, self.idx2char, num_generate=100)
        print(f"Seed: {self.seed_text}\nResult: {generated}\n")

def generate_text(model, start_string, char2idx, idx2char, num_generate=500, temperature=1.0):
    """Generates text using the trained model (stateless sliding window)."""
    input_text = start_string
    text_generated = []

    for i in range(num_generate):
        # Convert current context to indices
        # We handle <UNK> characters by skipping or erroring? 
        # For this demo we assume start_string is clean or caught in try/except in main
        input_eval = [char2idx[s] for s in input_text[-SEQ_LENGTH:]] # Use last 100
        
        input_tensor = tf.expand_dims(input_eval, 0)

        predictions = model(input_tensor)
        # predictions shape: (1, seq_len, vocab_size)
        # We need the last timestep
        predictions = predictions[:, -1, :]

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        
        predicted_char = idx2char[predicted_id]
        text_generated.append(predicted_char)
        input_text += predicted_char
        
    return (start_string + ''.join(text_generated))

def main():
    # 1. Download
    download_dataset(DATA_URL, LOCAL_FILE)
    
    # 2. Preprocess
    text, text_as_int, vocab, char2idx, idx2char = preprocess_text(LOCAL_FILE)
    
    if DEMO_MODE:
        # Reduce dataset size for quick check
        print("Running in DEMO MODE (reduced dataset size)")
        text_as_int = text_as_int[:100000] 

    # 3. specific Dataset Prep
    dataset = create_dataset(text_as_int, SEQ_LENGTH, BATCH_SIZE, BUFFER_SIZE)
    
    vocab_size = len(vocab)
    
    # 4. Model Design
    model = build_model(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        rnn_units=RNN_UNITS,
        batch_size=BATCH_SIZE)
    
    model.compile(optimizer='adam', loss=loss_function)
    
    # Directory for checkpoints
    checkpoint_dir = './training_checkpoints'
    # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    
    # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #    filepath=checkpoint_prefix,
    #    save_weights_only=True)
        
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    
    # Callback to print generated text during training
    text_gen_callback = TextGenerator(model, char2idx, idx2char)

    # 5. Model Training
    print("Starting training...")
    history = model.fit(dataset, 
                        epochs=EPOCHS, 
                        callbacks=[text_gen_callback])
    
    print("\nTraining complete.")
    
    # 6. Text Generation Final
    print("Generating final samples...")
    seeds = ["to be or not to be", "the king said", "where art thou"]
    for seed in seeds:
        print(f"\n--- Seed: {seed} ---")
        try:
            # Clean seed to ensure it only has valid chars if strictly enforcing
            clean_seed = seed.lower().translate(str.maketrans('', '', string.punctuation))
            # If our vocab doesn't include spaces (unlikely) we'd fail, but usually it does.
            # If seed contains chars not in vocab, we should handle or pick valid seeds.
            # Assuming vocab covers basic english chars.
            gen = generate_text(model, clean_seed, char2idx, idx2char)
            print(gen)
        except KeyError as e:
            print(f"Skipping seed '{seed}' due to character not in vocab: {e}")

    # Save the final model
    model.save('lstm_text_generator_model.h5')
    print("Model saved to lstm_text_generator_model.h5")

if __name__ == '__main__':
    main()
