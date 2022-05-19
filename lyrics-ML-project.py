# Tierney Salzmann

# This program will use the long short-term memory algorithm on the provided dataset (The Midnight song lyrics)
# to generate text/lyrics

# Import libraries
import numpy as np
import random
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


# Load text dataset, assign to raw_text variable, and convert all characters to lowercase 
file1 = r'C:\Users\NCG\Desktop\Projects\Machine Learning\lyrics.txt'
raw_text = open(file1, 'r', encoding = 'utf-8').read()
raw_text = raw_text.lower()


# Characters must be encoded as integers where each unique character is assigned an integer value
# List of every character
chars = sorted(list(set(raw_text)))
# Create dictionary of characters that map to respective integer values
char_to_int = dict((c, i) for i, c in enumerate(chars))
# Reverse map so text prediction prints in chars and not integers
int_to_char = dict((i, c) for i, c in enumerate(chars))


# Summarize dataset by printing total number of chars and total number of unique chars (vocab)
n_chars = len(raw_text)
n_vocab = len(chars)
print('Total number of characters: ', n_chars)
print('Total vocab: ', n_vocab)


# Now we must train the data by creating input/output sequences for LSTM
seq_length = 200 # Length of input sequences
sentences = [] # X value is sentences
next_chars = [] # Y value is next chars we are trying to predict/generate
for i in range(0, n_chars - seq_length, 1):
    sentences.append(raw_text[i: i + seq_length]) # Sequence in
    next_chars.append(raw_text[i + seq_length]) # Sequence out
n_patterns = len(sentences) # Define variable to display total number of patterns/sequences
print('Total number of sequences: ', n_patterns)

# Reshape input so it is compatible with LSTM: [samples, time steps, features]
# where time steps = sequence length and features = number of chars in vocab (n_vocab)
X = np.zeros((len(sentences), seq_length, n_vocab), dtype = np.bool)
Y = np.zeros((len(sentences), n_vocab), dtype = np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_int[char]] = 1
    Y[i, char_to_int[next_chars[i]]] = 1

print(X.shape)
print(Y.shape)

# Create and define our LSTM algorithm/model
# This is a basic model with only one LSTM
model = Sequential()
model.add(LSTM(256, input_shape = (seq_length, n_vocab)))
model.add(Dense(n_vocab, activation = 'softmax'))

optimizer = RMSprop(learning_rate = 0.01)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer)
model.summary()

# Create checkpoint 
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Fit our model with the data
model.fit(X, Y, epochs = 20, batch_size = 128, callbacks = callbacks_list)


# Define function to print char with max probability for our LSTM
# Probability array
def sample(preds):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Load checkpoint
filename = 'weights-improvement-16-0.4330.hdf5'
model.load_weights(filename)

# Start making predictions 
# Pick a random sentence from text as a seed sequence for our input
# Next char is then generated and updates the seed sequence
# Process is repeated for entire sequence length, in this case, 500 chars
start_index = random.randint(0, n_chars - seq_length - 1)
generated = ''
sentence = raw_text[start_index: start_index + seq_length]
generated += sentence

print('Seed for text prediction: "' + sentence + '"')

for i in range(1000):
    x_pred = np.zeros((1, seq_length, n_vocab))
    for t, char in enumerate(sentence):
        x_pred[0, t, char_to_int[char]] = 1

    preds = model.predict(x_pred, verbose = 0)[0]
    next_index = sample(preds)
    next_char = int_to_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char

    sys.stdout.write(next_char)
    sys.stdout.flush()

print()

