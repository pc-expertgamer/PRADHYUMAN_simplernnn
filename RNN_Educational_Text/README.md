## RNN for Educational Text Classification and Next Word Generation

This project implements Recurrent Neural Network (RNN) models using Keras (with a TensorFlow backend) to perform two Natural Language Processing (NLP) tasks:
1.  **Educational Text Classification**: Classifying short educational text snippets into predefined categories (Math, Science, History, English).
2.  **Next Word Generation**: Predicting the next 20 words given a starting sentence of at least 10 words.

This project was developed as a mandatory assignment to demonstrate understanding of RNN principles and implementation.

## Project Structure

RNNN_folder/   
├── README.md                # This file
├── datasets/                # Directory for input data
│   ├── classification_data.txt  # Labeled text snippets for classification
│   └── generation_data.txt    # Text corpus for next word generation
├── models/                  # Directory for saved trained models (created by scripts)
│   ├── classification_model.keras # Saved classification model
│   ├── classification_tokenizer.pkl # Saved tokenizer for classification
│   └── generation_model.keras   # Saved generation model (and potentially its tokenizer)
├── src/                     # Source code directory
│   ├── classification_model.py  # Script for classification task
│   └── generation_model.py    # Script for generation task (To be implemented)
└── utils/                   # Utility functions
    └── preprocessing.py       # Text preprocessing helper functions

## Requirements


    Python 3.8+

    TensorFlow 2.x (which includes Keras)

    NumPy

    Scikit-learn (for train_test_split)

Install required packages (preferably in a virtual environment):

bash
pip install tensorflow numpy scikit-learn

How to Run
1. Classification Model

To train and test the classification model:

bash
cd src
python classification_model.py

This will:

    Load and preprocess the classification dataset

    Train an LSTM model

    Evaluate the model on test data

    Save the trained model to models/classification_model.keras

    Test the model on sample sentences

2. Generation Model

To train and test the generation model:

bash
cd src
python generation_model.py

This will:

    Load and preprocess the generation dataset

    Train an LSTM model for next word prediction

    Save the trained model to models/generation_model.keras

    Generate text from sample seed sentences

Model Architecture
Classification Model

Input: Educational text snippets

Architecture:

    Embedding layer (64 dimensions)

    LSTM layer (64 units)

    Dropout (0.4)

    Dense layer (softmax activation)

Output: Category prediction (Math, Science, History, English)
Generation Model

Input: Sequence of words

Architecture:

    Embedding (100 dimensions)

    LSTM (256 units)

    Dropout (0.2)

    Dense (softmax, vocabulary size)

Output: Next word prediction
Dataset Creation

    classification_data.txt: Contains 100 short educational text snippets, with 25 samples for each of the 4 categories: Math, Science, History, and English. Each line follows the format text_snippet|Category.

    generation_data.txt: Contains a continuous text corpus of approximately 1200 words focused on Science. This corpus was designed to train the next-word prediction model and features coherent, grammatically correct sentence structures.

Sample Input and Output
Classification

Input:

text
The quadratic formula helps us solve equations of the form ax² + bx + c = 0

Output:

text
Predicted class: Math (confidence: 0.9245)

Text Generation

Input (seed text):

text
Science is the systematic study of the structure and behavior

Output:

text
Science is the systematic study of the structure and behavior of the physical and natural world through observation and experiment the scientific method is a fundamental approach used by scientists worldwide it begins with observations and questions followed by forming hypotheses making predictions and conducting experiments to test those predictions

Performance
Classification Task

    TensorFlow Version: 2.19.0

    Training samples: 80, Test samples: 20

    Effective vocabulary size: 873

    Final Test Accuracy: 80.00%

    Model Parameters: Embedding (873×64), LSTM (64 units), Dense (4 outputs)

Training Progress:

    Epoch 1: Training accuracy 24.84%, Validation accuracy 40.00%

    Epoch 15: Training accuracy 100.00%, Validation accuracy 80.00%

    Final test loss: 0.4711, Final test accuracy: 80.00%

Sample Predictions:

    Input Text: "Calculus is a fundamental concept in higher mathematics."
    Predicted Category: English (Confidence: 0.44)

    Input Text: "The water cycle describes the continuous movement of water on, above and below the surface of the Earth."
    Predicted Category: History (Confidence: 0.84)

    Input Text: "The Magna Carta was a charter of rights agreed to by King John of England in 1215."
    Predicted Category: History (Confidence: 0.99)

    Input Text: "Shakespearean sonnets are composed of 14 lines in iambic pentameter."
    Predicted Category: English (Confidence: 0.56)

Generation Task

    TensorFlow Version: 2.19.0

    Training sequences: 993, Validation sequences: 249

    Effective vocabulary size: 555

    Model: LSTM-based next word predictor with embedding and dropout

Output Example:

Input Seed:

text
Science is the systematic study of the structure and behavior

Generated Next Words:

text
of the physical and natural world through observation and experiment the scientific method is a fundamental approach used by scientists worldwide it begins with observations and questions followed by forming hypotheses making predictions and conducting experiments to test those predictions

Observations:

    The generated text is thematically coherent and maintains the scientific subject context.

    The model produces grammatically correct and contextually relevant continuations for educational text.

Key Implementation Notes

    Dataset Creation: Synthetic datasets for both tasks, containing educational content that's properly formatted and labeled.

    Preprocessing: Text cleaning, tokenization, sequence creation, and padding are handled by the preprocessing module.

    Model Architecture:

        Classification: Single LSTM layer for efficient learning on small dataset

        Generation: LSTM layers for sequence modeling

    Training Process:

        Model checkpointing to save the best version

        Training history monitoring

        Proper train/validation splits

    Temperature Parameter: The generation model includes a temperature parameter to control the randomness/creativity of the generated text.

Challenges & Key Learnings

    Dataset Significance: The quality, distinctiveness, and size of the training dataset are paramount for model performance, especially for NLP tasks.

    Overfitting: A primary challenge with small datasets, where models learn the training data too well but fail on unseen data. The classification model showed signs of overfitting with 100% training accuracy vs 80% test accuracy.

    Preprocessing Pipeline: Establishing a robust preprocessing pipeline is fundamental.

    RNN Fundamentals: Practical experience implementing RNNs with Keras, including Embedding layers, LSTM layers, Dropout, and Dense layers with softmax.

    Interpreting Metrics: Learned to analyze training/validation loss and accuracy curves to diagnose issues like overfitting.

    Iterative Process: Model development is iterative, involving data refinement, architecture adjustments, and hyperparameter tuning.

Future Improvements

    Implement attention mechanisms for better context understanding

    Use pre-trained word embeddings (GloVe, Word2Vec)

    Increase dataset size and diversity

    Experiment with transformer-based architectures

    Advanced generation sampling (temperature, top-k, nucleus)

    Early stopping based on validation loss
