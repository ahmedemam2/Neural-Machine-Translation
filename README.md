# Neural Machine Translation with Attention

## Overview
This project implements a sequence-to-sequence (seq2seq) neural machine translation (NMT) model with attention. The model consists of an encoder-decoder architecture using bidirectional LSTMs, enhanced with an attention mechanism to improve translation accuracy.

## Features
- **Encoder-Decoder Architecture**: Utilizes bidirectional LSTMs for encoding and a unidirectional LSTM for decoding.
- **Attention Mechanism**: Implements cross-attention to dynamically focus on relevant parts of the input sequence during translation.
- **Minimum Bayes Risk (MBR) Decoding**: Enhances translation quality by optimizing based on expected loss.

## Installation
To run this project, install the required dependencies:
```sh
pip install numpy
pip install tensorflow
```

## Usage
1. Prepare the dataset and preprocess input sequences.
2. Train the model using 'compile_and_train' function
3. Translate sentences using the 'mbr_decode' function

## Folder Structure
```
├── por.txt/       # Dataset and preprocessing scripts
├── utils.py/      # helper functions
├── nmt.py/        # Implementation of encoder, decoder, and attention
├── README.md      # Project documentation
```
