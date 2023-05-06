This script is designed to train a deep learning model for sentiment analysis on movie reviews. The model classifies a given movie review as either positive or negative. The dataset used for this task is the IMDB movie reviews dataset.


You can customize the scripts by modifying the following hyperparameters:
- vocab_size: The maximum number of words to keep in the vocabulary, based on word frequency.
- max_len: The maximum length of the tokenized and padded reviews.
- You can also modify the model architecture by adding or changing layers in the create_model function.

The kFold script approaches the problem using k-fold validation where you can calso customize the following hyperparameters
- embedding_dim: The size of the word embeddings.
- k: The number of folds for k-fold cross-validation.
