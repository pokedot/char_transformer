# Basic Transformer For next character prediction

This repository contains a minimal character-level Transformer (decoder-only) implemented in JAX/Flax for next-character prediction, and also an LSTM model to be used as a baseline comparison. 

Repository structure
--------------------

```
char_transformer/
  data/                       <- Preprocessed text8 datasets used for the project  
    text8_test.txt
    text8_train.txt
  models/
    models.py                 <- Minimal, decoder-only Transformer implementation
  results/
  util/
    generation.py             <- Autoregressive token generator for the trained Flax/JAX model
  LSTM_auto_reduce_lr.ipynb   <- LSTM model used as a baseline for comparison
  LSTM_text8.ipynb
  requirements.txt            <- A list of all packages required for this project
  transformer.ipynb           <- Primary Jupyter notebook used for experimenting, training and generation
```

Getting started
------------------
1. Install the required packages in a terminal before running the notebooks: `pip install -r requirements.txt`
2. Run the `transformer.ipynb` notebook!