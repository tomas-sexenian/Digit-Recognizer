import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping

# Load the data
vocab_size = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Preprocess the text data
word_index = imdb.get_word_index()
index_to_word = {index + 3: word for word, index in word_index.items()}
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<UNK>"
index_to_word[3] = "<UNUSED>"

def preprocess_text(text):
    text = ' '.join(text)
    text = text.lower()
    tokenized_text = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokenized_text = [word for word in tokenized_text if word.isalnum() and word not in stop_words]
    return ' '.join(tokenized_text)

x_train_text = [' '.join([index_to_word[i] for i in review]) for review in x_train]
x_test_text = [' '.join([index_to_word[i] for i in review]) for review in x_test]

x_train_preprocessed = [preprocess_text(review) for review in x_train_text]
x_test_preprocessed = [preprocess_text(review) for review in x_test_text]

# Tokenize and pad the text data
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x_train_preprocessed)

x_train_tokenized = tokenizer.texts_to_sequences(x_train_preprocessed)
x_test_tokenized = tokenizer.texts_to_sequences(x_test_preprocessed)

max_len = 100
x_train_padded = pad_sequences(x_train_tokenized, maxlen=max_len)
x_test_padded = pad_sequences(x_test_tokenized, maxlen=max_len)

# Define the model architecture
embedding_dim = 20
def create_model():
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(Bidirectional(LSTM(32, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))  # Reduced number of units
    model.add(Bidirectional(LSTM(16, dropout=0.5, recurrent_dropout=0.5)))  # Reduced number of units
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(lr=0.001), loss=BinaryCrossentropy(), metrics=['accuracy'])
    return model

# Perform k-fold cross-validation
k = 3
kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

val_loss_per_fold = []

for train_indices, val_indices in kfold.split(x_train_padded, y_train):
    train_data, train_labels = x_train_padded[train_indices], y_train[train_indices]
    val_data, val_labels = x_train_padded[val_indices], y_train[val_indices]
    
    model = create_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), batch_size=64, epochs=10, callbacks=[early_stopping])
    
    val_loss_per_fold.append(min(history.history['val_loss']))

# Calculate the average validation loss
avg_val_loss = np.mean(val_loss_per_fold)
print(f'Average validation loss after k-fold cross-validation: {avg_val_loss}')

# Train the final model on the entire training dataset
model = create_model()
history = model.fit(x_train_padded, y_train, validation_data=(x_test_padded, y_test), batch_size=64, epochs=10, callbacks=[early_stopping])

# Plot the mean absolute error
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make a prediction
def predict_review(text):
    preprocessed_text = preprocess_text(text)
    tokenized_text = tokenizer.texts_to_sequences([preprocessed_text])
    padded_text = pad_sequences(tokenized_text, maxlen=max_len)
    prediction = model.predict(padded_text)
    return "Positive" if prediction >= 0.5 else "Negative"

user_review = input("Enter a movie review: ")
print("The model's prediction: {predict_review(user_review)}")