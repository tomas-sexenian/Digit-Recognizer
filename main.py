import numpy as np
import matplotlib.pyplot as plt
import string
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, GRU, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Reshape

nltk.download('stopwords')

vocab_size = 10000
max_len = 500

# Preprocess the text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    translator = str.maketrans('', '', string.punctuation)
    words = text.lower().translate(translator).split()
    return ' '.join([word for word in words if word not in stop_words])

# Load the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Get the word index and reverse it
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

# Convert integer sequences to text
x_train_text = [' '.join([reverse_word_index.get(i - 3, '') for i in sequence]) for sequence in x_train]
x_test_text = [' '.join([reverse_word_index.get(i - 3, '') for i in sequence]) for sequence in x_test]

# Preprocess the data
x_train_text = [preprocess_text(text) for text in x_train_text]
x_test_text = [preprocess_text(text) for text in x_test_text]

# Load pretrained GloVe embeddings from TensorFlow Hub
embedding_dim = 20
embedding_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1", output_shape=[embedding_dim], input_shape=[], dtype=tf.string, trainable=False)

# Build the deep neural network
model = Sequential()
model.add(embedding_layer)
model.add(Reshape((1, embedding_dim)))
model.add(Bidirectional(GRU(64, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
model.add(Bidirectional(GRU(32, dropout=0.5, recurrent_dropout=0.5)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss=BinaryCrossentropy(), metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(np.array(x_train_text), y_train, validation_data=(np.array(x_test_text), y_test), batch_size=64, epochs=10, callbacks=[early_stopping])

# Plot MAE
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.show()

# Make a prediction for the user input
def predict_review(review):
    review = preprocess_text(review)
    review = np.array([review])
    prediction = model.predict(review)
    return 'Positive' if prediction > 0.5 else 'Negative'

user_review = input("Enter a movie review: ")
print("The model's prediction:", predict_review(user_review))