# utils/preprocessing.py
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def clean_text(text):
    """Cleans text by lowercasing, removing punctuation, and normalizing whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Keep only word characters and whitespace
    text = re.sub(r'\s+', ' ', text)     # Replace multiple spaces with a single one
    return text.strip()

def preprocess_texts_for_classification(
    texts_list,
    labels_str_list,
    label_to_int_map,
    num_classes,
    max_vocab_size,
    max_sequence_length,
    tokenizer_object=None
):
    """
    Preprocesses a list of texts and their string labels for a classification task.
    - Cleans text.
    - Tokenizes text (fits a new tokenizer if one isn't provided).
    - Converts text to padded sequences of integers.
    - Converts string labels to one-hot encoded vectors.
    
    Returns:
        padded_sequences (np.array): Processed input data.
        one_hot_labels (np.array): Processed label data.
        tokenizer (Tokenizer): The fitted tokenizer (either new or the one passed in).
        effective_vocab_size (int): The vocabulary size to be used for the Embedding layer.
    """
    print("Starting text preprocessing for classification...")

    # 1. Clean Texts
    cleaned_texts = [clean_text(text) for text in texts_list]
    print(f"Cleaned {len(cleaned_texts)} texts.")

    # 2. Tokenize Texts
    if tokenizer_object is None:
        print("No tokenizer provided, fitting a new one.")
        tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<UNK>")
        tokenizer.fit_on_texts(cleaned_texts)
    else:
        print("Using provided tokenizer.")
        tokenizer = tokenizer_object
    
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    print(f"Converted texts to {len(sequences)} sequences of integers.")

    # 3. Pad Sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    print(f"Padded sequences to shape: {padded_sequences.shape}")

    # 4. Process Labels
    try:
        numeric_labels = [label_to_int_map[label] for label in labels_str_list]
    except KeyError as e:
        print(f"FATAL ERROR: Unknown label encountered during mapping: {e}. Ensure all labels in data are in label_to_int_map.")
        sys.exit(1)
        
    one_hot_labels = to_categorical(np.array(numeric_labels), num_classes=num_classes)
    print(f"One-hot encoded labels to shape: {one_hot_labels.shape}")
    
    effective_vocab_size = min(max_vocab_size, len(tokenizer.word_index) + 1)
    if effective_vocab_size <= 1 and len(tokenizer.word_index) > 0 : # Check for edge case where max_vocab_size might be too small
        effective_vocab_size = len(tokenizer.word_index) + 1 # Ensure at least some vocab if words exist
    print(f"Effective vocabulary size for model: {effective_vocab_size}")

    print("Text preprocessing for classification complete.")
    return padded_sequences, one_hot_labels, tokenizer, effective_vocab_size
# --- NEW FUNCTION FOR GENERATION PREPROCESSING ---
def preprocess_corpus_for_next_word_prediction(
    corpus_text, 
    seq_len, 
    tokenizer_to_fit=None, 
    max_vocab_size=None, 
    oov_token_char="<UNK>"
):
    """
    Cleans, tokenizes a text corpus, and creates input sequences and corresponding 
    target words for training a next-word prediction model.

    Args:
        corpus_text (str): The raw text corpus.
        seq_len (int): The length of the input sequences (e.g., 10 words).
        tokenizer_to_fit (Tokenizer, optional): An existing Keras Tokenizer. 
                                                If None, a new one is created and fitted.
        max_vocab_size (int, optional): Maximum vocabulary size for the tokenizer. If None, no limit.
        oov_token_char (str): Token to use for out-of-vocabulary words.

    Returns:
        X (np.array): Numpy array of input sequences (integer encoded).
        y (np.array): Numpy array of target words (one-hot encoded).
        tokenizer (Tokenizer): The fitted Keras Tokenizer.
        effective_vocab_size_gen (int): The actual vocabulary size used for the model.
    """
    print("\nStarting corpus preprocessing for next-word prediction...")

    cleaned_corpus = clean_text(corpus_text)
    print(f"Cleaned corpus. New length: {len(cleaned_corpus)} characters.")

    if tokenizer_to_fit is None:
        print("No tokenizer provided, fitting a new one on the generation corpus.")
        tokenizer = Tokenizer(num_words=max_vocab_size, oov_token=oov_token_char, char_level=False)
        tokenizer.fit_on_texts([cleaned_corpus])
    else:
        print("Using provided tokenizer.")
        tokenizer = tokenizer_to_fit
    
    corpus_tokens = tokenizer.texts_to_sequences([cleaned_corpus])[0]
    print(f"Corpus tokenized into {len(corpus_tokens)} tokens.")

    if len(corpus_tokens) <= seq_len:
        print(f"FATAL ERROR: Corpus has {len(corpus_tokens)} tokens, which is not enough "
              f"to create sequences of length {seq_len}. Need at least {seq_len + 1} tokens.")
        sys.exit(1)

    input_sequences = []
    target_word_indices = []
    for i in range(len(corpus_tokens) - seq_len):
        input_seq = corpus_tokens[i : i + seq_len]
        target_word_idx = corpus_tokens[i + seq_len]
        input_sequences.append(input_seq)
        target_word_indices.append(target_word_idx)
    
    print(f"Created {len(input_sequences)} input/target sequences.")

    X = np.array(input_sequences)
    
    if max_vocab_size:
        effective_vocab_size_gen = min(max_vocab_size, len(tokenizer.word_index) + 1)
    else:
        effective_vocab_size_gen = len(tokenizer.word_index) + 1
    
    if effective_vocab_size_gen <= 1 and len(tokenizer.word_index) > 0:
        effective_vocab_size_gen = len(tokenizer.word_index) + 1
        print(f"Adjusted effective_vocab_size_gen due to low value: {effective_vocab_size_gen}")


    print(f"Effective vocabulary size for generation model (for one-hot encoding y): {effective_vocab_size_gen}")

    y = to_categorical(np.array(target_word_indices), num_classes=effective_vocab_size_gen)
    
    print(f"Input sequences (X) shape: {X.shape}")
    print(f"Target words (y) shape: {y.shape}")
    print("Corpus preprocessing for next-word prediction complete.")
    
    return X, y, tokenizer, effective_vocab_size_gen