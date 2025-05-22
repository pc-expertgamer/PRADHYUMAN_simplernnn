# src/classification_model.py

import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

print(f"TensorFlow Version: {tf.__version__}")

# Path setup for local utils import
try:
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    if project_root not in sys.path:
        sys.path.append(project_root)
    from utils.preprocessing import clean_text, preprocess_texts_for_classification
except ImportError as e:
    print(f"Error importing from utils: {e}")
    sys.exit(1)

print("Imports and path setup complete.")

# Configuration
_current_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(_current_script_dir)

CLASSIFICATION_DATA_PATH = os.path.join(PROJECT_ROOT_DIR, 'datasets', 'classification_data.txt')
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT_DIR, 'models')
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'classification_model.keras')

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Model hyperparameters
MAX_VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 64
RNN_UNITS = 64
DROPOUT_RATE = 0.4
NUM_CLASSES = 4
LEARNING_RATE = 0.001

# Training parameters
EPOCHS = 15
BATCH_SIZE = 16
TEST_SPLIT_SIZE = 0.2

# Label mappings
LABEL_TO_INT_MAP = {"Math": 0, "Science": 1, "History": 2, "English": 3}
INT_TO_LABEL_MAP = {v: k for k, v in LABEL_TO_INT_MAP.items()}

print("Configuration constants defined.")
print(f"Data path: {CLASSIFICATION_DATA_PATH}")
print(f"Model save path: {MODEL_SAVE_PATH}")

def load_classification_data(file_path):
    """Load text snippets and labels from file. Format: 'text|Category'"""
    texts = []
    labels_str_list = []
    
    print(f"Attempting to load data from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('|', 1)
                if len(parts) == 2:
                    text, label_str = parts[0].strip(), parts[1].strip()
                    if label_str in LABEL_TO_INT_MAP:
                        texts.append(text)
                        labels_str_list.append(label_str)
                    else:
                        print(f"Warning: Line {line_num}: Unknown label '{label_str}'. Skipping.")
                else:
                    print(f"Warning: Line {line_num}: Malformed line: '{line}'. Skipping.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Dataset file not found at {file_path}")
        sys.exit(1)
    
    if not texts:
        print(f"FATAL ERROR: No valid data loaded from {file_path}")
        sys.exit(1)
        
    print(f"Successfully loaded {len(texts)} text samples.")
    return texts, labels_str_list

def build_classification_model(vocab_size_for_embedding, num_classes_for_output):
    """Build and compile RNN model for text classification"""
    print(f"\nBuilding classification model...")
    print(f"  Vocabulary size: {vocab_size_for_embedding}")
    print(f"  Embedding dimension: {EMBEDDING_DIM}")
    print(f"  Max sequence length: {MAX_SEQUENCE_LENGTH}")
    print(f"  RNN units: {RNN_UNITS}")
    print(f"  Dropout rate: {DROPOUT_RATE}")
    print(f"  Number of output classes: {num_classes_for_output}")

    model = Sequential(name="Simple_RNN_Text_Classifier")

    # Embedding layer: converts integer tokens to dense vectors
    model.add(Embedding(input_dim=vocab_size_for_embedding,
                        output_dim=EMBEDDING_DIM,
                        input_length=MAX_SEQUENCE_LENGTH,
                        mask_zero=True,
                        name="embedding_layer"))

    # LSTM layer: processes sequence of embeddings
    model.add(LSTM(units=RNN_UNITS, name="lstm_layer"))

    # Dropout for regularization
    model.add(Dropout(rate=DROPOUT_RATE, name="dropout_layer"))

    # Output layer: softmax for multi-class classification
    model.add(Dense(units=num_classes_for_output, activation='softmax', name="output_layer"))

    print("\nCompiling the model...")
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("\nModel Summary:")
    model.summary()
    
    print("\nClassification model built and compiled successfully.")
    return model

def main():
    """Main pipeline: load, preprocess, train, evaluate, and save model"""
    print("--- Starting Educational Text Classification Task ---")

    # Load data
    raw_texts, raw_labels_str = load_classification_data(CLASSIFICATION_DATA_PATH)
    
    # Preprocess data
    X_processed, y_processed, fitted_tokenizer, effective_vocab_size = \
        preprocess_texts_for_classification(
            texts_list=raw_texts,
            labels_str_list=raw_labels_str,
            label_to_int_map=LABEL_TO_INT_MAP,
            num_classes=NUM_CLASSES,
            max_vocab_size=MAX_VOCAB_SIZE,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            tokenizer_object=None 
        )

    # Check dataset size
    if X_processed.shape[0] < NUM_CLASSES / (1-TEST_SPLIT_SIZE) and X_processed.shape[0] < NUM_CLASSES / TEST_SPLIT_SIZE:
         print(f"Warning: Dataset size ({X_processed.shape[0]}) is very small for {NUM_CLASSES} classes.")

    # Split data
    print("\nSplitting data into training and testing sets...")
    stratify_labels_int = np.array([LABEL_TO_INT_MAP[label] for label in raw_labels_str])

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed,
            test_size=TEST_SPLIT_SIZE,
            random_state=42,
            stratify=stratify_labels_int 
        )
    except ValueError as e:
        print(f"FATAL ERROR during train_test_split: {e}")
        from collections import Counter
        print(f"Label distribution: {Counter(stratify_labels_int)}")
        sys.exit(1)
        
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    # Build model
    model = build_classification_model(effective_vocab_size, NUM_CLASSES)

    # Train model
    print("\nTraining the model...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Evaluate model
    print("\nEvaluating the model on the test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss: {loss:.4f}")
    print(f"  Test Accuracy: {accuracy:.4f} ({(accuracy*100):.2f}%)")

    # Save model and tokenizer
    print(f"\nSaving the trained model to: {MODEL_SAVE_PATH}")
    try:
        model.save(MODEL_SAVE_PATH)
        print("  Model saved successfully.")
    except Exception as e:
        print(f"  Error saving model: {e}")

    import pickle
    tokenizer_save_path = os.path.join(MODEL_SAVE_DIR, 'classification_tokenizer.pkl')
    try:
        with open(tokenizer_save_path, 'wb') as handle:
            pickle.dump(fitted_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Tokenizer saved successfully to: {tokenizer_save_path}")
    except Exception as e:
        print(f"  Error saving tokenizer: {e}")

    # Demonstrate predictions
    print("\n--- Demonstrating Classification on Sample Unseen Texts ---")
    sample_texts_for_demo = [
        "Calculus is a fundamental concept in higher mathematics.",
        "The water cycle describes the continuous movement of water on, above and below the surface of the Earth.",
        "The Magna Carta was a charter of rights agreed to by King John of England in 1215.",
        "Shakespearean sonnets are composed of 14 lines in iambic pentameter."
    ]

    for text_input in sample_texts_for_demo:
        # Preprocess single text sample
        cleaned_input = clean_text(text_input)
        sequence = fitted_tokenizer.texts_to_sequences([cleaned_input])
        padded_input = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        
        # Make prediction
        prediction_probs = model.predict(padded_input, verbose=0)[0]
        predicted_class_index = np.argmax(prediction_probs)
        predicted_class_label = INT_TO_LABEL_MAP[predicted_class_index]
        confidence = prediction_probs[predicted_class_index]
        
        print(f"\nInput Text: \"{text_input}\"")
        print(f"  Predicted Category: {predicted_class_label} (Confidence: {confidence:.2f})")

    print("\n--- Classification Task Completed ---")

if __name__ == "__main__":
    main()
