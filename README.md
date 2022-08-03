# numpy-RNN
LSTM-RNN Implementation w/ NumPy


#### WHERE THE PROJECT IS AT:
* Create Vocab
* Create Embedding Layer
* Create LSTM Layer
* Perform Forward Pass

-----

#### EMBEDDING: NUMPY VS KERAS IMPLEMENTATION
* NUMPY:
```
## create vocabulary + tokenize
vocab = Vocabulary()
token_sequences = vocab.tokenize(data, 26)

## create embedding layer
embedding = EmbeddingLayer(vocab_size=vocab.size, hidden_dim=50) ## hidden_dim is a hyper-param

## create X & Y datasets
X = token_sequences[:,:-1]
y = token_sequences[:,-1]

lstm_inputs = embedding.predict(X)
print(lstm_inputs.shape) ## batch_size x seq_length x dimensionality
```

* Keras:
```
from keras.preprocessing.text import Tokenizer

t  = Tokenizer()
t.fit_on_texts(data)
sequences = t.texts_to_sequences(data)

## create X & Y datasets
X = sequences[:,:-1]
y = sequences[:,-1]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, hidden_dim, seq_length))

model.compile('rmsprop', 'mse')
lstm_inputs = model.predict(X)
print(lstm_inputs.shape) ## batch_size x seq_length x dimensionality
```

-----

#### LSTM: NUMPY VS KERAS IMPLEMENTATION
* NUMPY:
```
from layers.lstm import LSTM

## these are all hyper-params that were defined during tokenizing/embedding
## we are explicitly defining them for this example but would need to come from
## the tokenizer and embedding layer attributes during a training run
HIDDEN = 100
FEATURES = 20
BATCH_SIZE = 500
SEQ_LENGTH = 12

inputs = np.random.randn(BATCH_SIZE, SEQ_LENGTH, FEATURES) ## in training this is embedding output!
lstm = LSTM(units=HIDDEN, features=FEATURES)

out_last = lstm.forward(inputs) ## batch_size x hidden
out_full = lstm.forward(inputs, return_sequences=True) ## batch_size x seq_length x hidden
```

* Keras:
```
## hyper-params (discussed above)
HIDDEN = 100
FEATURES = 20
BATCH_SIZE = 500
SEQ_LENGTH = 12

inputs = np.random.randn(BATCH_SIZE, SEQ_LENGTH, FEATURES) ## in training this is embedding output!

klstm = tf.keras.layers.LSTM(HIDDEN, return_sequences=True)
out_full = klstm(inputs)

klstm1 = tf.keras.layers.LSTM(HIDDEN)
out_last = klstm1(inputs)
```
