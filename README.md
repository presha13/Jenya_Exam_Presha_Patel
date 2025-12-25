# LSTM Text Generator

This project implements a character-level LSTM text generation model using TensorFlow/Keras. It is trained on Shakespeare's Complete Works (downloaded automatically).

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Script**:
   ```bash
   python lstm_text_generator.py
   ```

   The script will:
   - Download `shakespeare.txt`.
   - Preprocess the text (lowercase, remove punctuation, tokenize).
   - Train an LSTM model.
   - Generate text samples after each epoch.
   - Save the final model to `lstm_text_generator_model.h5`.

## Configuration

You can adjust hyperparameters in `lstm_text_generator.py`:
- `EPOCHS`: Number of training epochs (Default: 30).
- `RNN_UNITS`: Size of LSTM layer (Default: 1024).
- `DEMO_MODE`: Set to `True` for a quick test run with reduced data.

## Model Architecture
- **Embedding Layer**: Maps characters to dense vectors.
- **LSTM Layer**: 1024 units, returns sequences.
- **Dense Layer**: Predicting next character logits.

## Sample Output
(After training)
Seed: "shall i compare thee"
Result: "shall i compare thee to a summers day thou art more lovely and more temperate rough winds do shake the darling buds of..."
