# src/generation_model.py

import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

print(f"TensorFlow Version: {tf.__version__}")

# --- Import local preprocessing utilities ---
try:
    _current_script_dir = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT_DIR = os.path.dirname(_current_script_dir)
    if PROJECT_ROOT_DIR not in sys.path:
        sys.path.append(PROJECT_ROOT_DIR)
    from utils.preprocessing import clean_text, preprocess_corpus_for_next_word_prediction
except ImportError as e:
    print(f"Error importing from utils: {e}")
    sys.exit(1)

# --- Paths and Hyperparameters ---
GENERATION_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, 'datasets', 'generation_data.txt')
MODEL_SAVE_DIR_GEN = os.path.join(PROJECT_ROOT_DIR, 'models')
MODEL_SAVE_PATH_GEN = os.path.join(MODEL_SAVE_DIR_GEN, 'generation_model.keras')
TOKENIZER_SAVE_PATH_GEN = os.path.join(MODEL_SAVE_DIR_GEN, 'generation_tokenizer.pkl')
os.makedirs(MODEL_SAVE_DIR_GEN, exist_ok=True)

MAX_VOCAB_SIZE_GEN = None
SEQ_LENGTH = 10
EMBEDDING_DIM_GEN = 100
RNN_UNITS_GEN = 256
DROPOUT_RATE_GEN = 0.2
EPOCHS_GEN = 50
BATCH_SIZE_GEN = 64
LEARNING_RATE_GEN = 0.001

def load_generation_corpus(file_path):
    print(f"Attempting to load generation corpus from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            corpus = f.read()
        if not corpus.strip():
            print(f"ERROR: Generation corpus file at {file_path} is empty.")
            sys.exit(1)
        print(f"Successfully loaded generation corpus. Length: {len(corpus)} characters.")
        return corpus
    except FileNotFoundError:
        print(f"ERROR: Generation corpus file not found at {file_path}")
        sys.exit(1)

def build_generation_model(vocab_size, seq_len):
    print(f"\nBuilding text generation model...")
    print(f"  Vocabulary size (for Embedding input_dim & Dense output_dim): {vocab_size}")
    print(f"  Sequence length (for Embedding input_length): {seq_len}")
    print(f"  Embedding dimension: {EMBEDDING_DIM_GEN}")
    print(f"  RNN units: {RNN_UNITS_GEN}")
    print(f"  Dropout rate: {DROPOUT_RATE_GEN}")

    model = Sequential(name="Simple_RNN_Text_Generator")
    model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM_GEN, input_length=seq_len, name="embedding_layer_gen"))
    model.add(LSTM(units=RNN_UNITS_GEN, name="lstm_layer_gen"))
    model.add(Dropout(rate=DROPOUT_RATE_GEN, name="dropout_layer_gen"))
    model.add(Dense(units=vocab_size, activation='softmax', name="output_layer_gen"))

    print("\nCompiling the generation model...")
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_GEN), loss='categorical_crossentropy', metrics=['accuracy'])

    print("\nGeneration Model Summary:")
    model.summary()
    print("\nText generation model built and compiled successfully.")
    return model

def generate_next_n_words(model, tokenizer, seed_text, num_words, max_sequence_len):
    print(f"\nGenerating {num_words} words from seed: '{seed_text}'")
    generated = ""
    input_text = seed_text.lower()
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        padded = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre', truncating='pre')
        predicted_probs = model.predict(padded, verbose=0)[0]
        predicted_token = np.argmax(predicted_probs)
        output_word = tokenizer.index_word.get(predicted_token, "<UNK>")
        if output_word == "<UNK>" or not output_word:
            print(f"  Generated '<UNK>' or empty word. Stopping generation.")
            break
        generated += " " + output_word
        input_text += " " + output_word
    return generated.strip()

def main():
    print("--- Starting Next Word Generation Task ---")
    raw_corpus = load_generation_corpus(GENERATION_DATA_PATH)
    X, y, tokenizer, vocab_size = preprocess_corpus_for_next_word_prediction(
        corpus_text=raw_corpus,
        seq_len=SEQ_LENGTH,
        tokenizer_to_fit=None,
        max_vocab_size=MAX_VOCAB_SIZE_GEN,
        oov_token_char="<UNK>"
    )
    if X.size == 0:
        print("ERROR: No sequences generated from corpus.")
        sys.exit(1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training sequences: {X_train.shape[0]}, Validation sequences: {X_val.shape[0]}")
    print(f"y_train shape: {y_train.shape}, y_val shape: {y_val.shape}")

    model = build_generation_model(vocab_size, SEQ_LENGTH)

    print("\nTraining the generation model...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS_GEN,
        batch_size=BATCH_SIZE_GEN,
        validation_data=(X_val, y_val),
        verbose=1
    )

    final_training_loss = history.history['loss'][-1]
    final_training_accuracy = history.history['accuracy'][-1]
    final_val_loss = history.history.get('val_loss', [float('nan')])[-1]
    final_val_accuracy = history.history.get('val_accuracy', [float('nan')])[-1]

    print("\n--- Final Model Metrics (Next Word Prediction Task) ---")
    print(f"  Final Training Loss: {final_training_loss:.4f}")
    print(f"  Final Training Accuracy: {final_training_accuracy:.4f} ({(final_training_accuracy*100):.2f}%)")
    print(f"  Final Validation Loss: {final_val_loss:.4f}")
    print(f"  Final Validation Accuracy: {final_val_accuracy:.4f} ({(final_val_accuracy*100):.2f}%)")

    print(f"\nSaving the trained generation model to: {MODEL_SAVE_PATH_GEN}")
    try:
        model.save(MODEL_SAVE_PATH_GEN)
        print("  Generation model saved successfully.")
    except Exception as e:
        print(f"  Error saving generation model: {e}")

    try:
        with open(TOKENIZER_SAVE_PATH_GEN, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Generation tokenizer saved successfully to: {TOKENIZER_SAVE_PATH_GEN}")
    except Exception as e:
        print(f"  Error saving generation tokenizer: {e}")

    print("\n--- Demonstrating Next Word Generation ---")
    seed_text = "The study of biology helps us understand the complex interactions"
    if len(clean_text(seed_text).split()) < SEQ_LENGTH:
        print(f"Warning: Seed text is shorter than SEQ_LENGTH ({SEQ_LENGTH}). Results might be poor.")
    num_words_to_generate = 20
    generated_words = generate_next_n_words(
        model, tokenizer, seed_text, num_words_to_generate, SEQ_LENGTH
    )
    print(f"\nSeed Text: \"{seed_text}\"")
    print(f"Generated Next {num_words_to_generate} Words: \"{generated_words}\"")
    print(f"Full Generated Text: \"{seed_text} {generated_words}\"")

    print("\nBriefly comment on the coherence of the generated text (manual assessment):")
    print("  (e.g., Does it make sense? Is it grammatically plausible? Does it stay on topic with the seed?)")

    print("\n--- Next Word Generation Task Completed ---")

if __name__ == "__main__":
    main()
