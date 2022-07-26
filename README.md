# numpy-RNN
**LSTM-RNN Implementation w/ NumPy**


## WHERE THE PROJECT IS AT
* Create Vocab
* Create Embedding Layer
* Create LSTM Layer
* Perform Forward Pass
* Perform Backward Pass

## Next Steps
* Different optimizers/loss functions/activation functions, etc...
* Right now the 'LSTMSequential' class is sort of hard coded in nature, meaning it needs to be in the form of Embedding -> LSTM -> Dense
* I want to abstract this even more so we can stack LSTMs, Dense layers, utilize pre-trained embeddings, etc..

## Notes
* Attempts to mimic the keras implementation
* Process: Tokenizer --> Embedding --> LSTM --> Dense
* Backpropagates from Dense all the way through Embeddings
* I encourage you to read the code for a deeper understanding, it is well documented

## Repo Structure
* Testing and development can be found in the 'notebooks' folder
* The tokenizer can be found in the 'utils' folder
* The embedding, lstm, and dense layers can be found in the 'layers' folder
* Unit tests are in the lstm_testing.py file. ```>>> pytest lstm_testing.py```

## README
1. **Implementation Overview**
2. **Embedding:** NumPy vs Keras
3. **LSTM:** NumPy vs Keras
4. **Usage**
-----

## Implementation

* This implementation follows the keras implementation 1 (there are two), where the hidden state is multiplied with a recurrent weight kernel denoted by 'U'
* Implementation 2 concatenates the inputs, denoted by 'X', with the hidden state 'h', into term 'z', then multiplies this with the weight kernel. A lot of publications I've seen online utilize implementatin 2.
* BPTT is computed after each batch, as opposed to forward-prop through the entire dataset and then back-prop through the entire dataset in reverse.

<br>

**Equations for implementation 1:**

<img src="https://latex.codecogs.com/svg.latex?f_%7Bt%7D%3D%5Csigma%28W_%7Bf%7DX_%7Bt%7D%2BU_%7Bf%7Dh_%7Bt-1%7D%2Bb_%7Bf%7D%29" height="100" width="450">

<img src="https://latex.codecogs.com/svg.latex?i_%7Bt%7D%3D%5Csigma%28W_%7Bi%7DX_%7Bt%7D%2BU_%7Bi%7Dh_%7Bt-1%7D%2Bb_%7Bi%7D%29" height="100" width="450">

<img src="https://latex.codecogs.com/svg.latex?o_%7Bt%7D%3D%5Csigma%28W_%7Bo%7DX_%7Bt%7D%2BU_%7Bo%7Dh_%7Bt-1%7D%2Bb_%7Bo%7D%29" height="100" width="450">

<img src="https://latex.codecogs.com/svg.latex?%5Cbar%7Bc%7D%3D%5Csigma%28W_%7Bc%7DX_%7Bt%7D%2BU_%7Bc%7Dh_%7Bt-1%7D%2Bb_%7Bc%7D%29" height="100" width="450">

<img src="https://latex.codecogs.com/svg.latex?c_%7Bt%7D%3Df_%7Bt%7D%2Ac_%7Bt-1%7D%2Bi_%7Bt%7D%2A%5Cbar%7Bc_%7Bt%7D%7D" height="100" width="450">

<img src="https://latex.codecogs.com/svg.latex?h_%7Bt%7D%3Do_%7Bt%7D%2Atanh%28c_%7Bt%7D%29" height="100" width="425">


-----
## **EMBEDDING:** NUMPY VS KERAS IMPLEMENTATION
* **NUMPY:**
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

* **Keras:**
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

## **LSTM:** NUMPY VS KERAS IMPLEMENTATION
* **NUMPY:**
```
from layers.lstm import LSTM

## these are all hyper-params that were defined during tokenizing/embedding (except the LSTM units)
## we are explicitly defining them for this example but would need to come from
## the tokenizer and embedding layer attributes during a training run

# note - the lstm 'forward' method expects a single batch of t_steps x embedding_dim. this method will be used in the 'Sequential' class

lstm = LSTM(units=100, features=20, seq_length=25)
init_state = {'h':np.zeros((100,)), 'c':np.zeros((100,))}
cache, state = lstm.forward(batch1, init_state)
```

* **Keras:**
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
-----
## **Usage**

**NumPy:**

```
from lstm import LSTM
from tokenizer import Vocabulary
from dense import Dense
from embedding import EmbeddingLayer
from sequential import LSTMSequential

# step 1 -- data
f = open(r"<path/to/data.txt>", 'r', encoding='utf-8').readlines()

# step 2 -- tokenize
## create vocabulary + tokenize
v = Vocabulary()
token_sequences = v.tokenize(f, 26)

# step 3 -- split into x/y
## create X & Y datasets
X = token_sequences[:,:-1]
y = token_sequences[:,-1]

model = LSTMSequential()

model.add(EmbeddingLayer(vocab_size=v.size, hidden_dim=20))
model.add(LSTM(units=100, features=20, seq_length=25))
model.add(Dense(v.size, 100))

model.train(X, y, 200)

LOSS: 8.168681714555001, EPOCH: 0
LOSS: 8.087044925195558, EPOCH: 1
LOSS: 8.079370350953514, EPOCH: 2
LOSS: 8.073516901760831, EPOCH: 3
LOSS: 8.067811866708043, EPOCH: 4
LOSS: 8.056927123363462, EPOCH: 5
LOSS: 8.02688694851102, EPOCH: 6
LOSS: 7.9702779955759055, EPOCH: 7
LOSS: 7.862013461804196, EPOCH: 8
LOSS: 7.684731973489584, EPOCH: 9
LOSS: 7.432876064202686, EPOCH: 10
LOSS: 7.13236489440936, EPOCH: 11
LOSS: 6.832459447309209, EPOCH: 12
etc...
```