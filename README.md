# numpy-RNN
**LSTM-RNN Implementation w/ NumPy**


## WHERE THE PROJECT IS AT
* Create Vocab
* Create Embedding Layer
* Create LSTM Layer
* Perform Forward Pass
* Perform Backward Pass

## Next Steps
* Develop 'Sequential' object to train (using SGD)
* Different optimizers/loss functions, etc...

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
1. **Implementation**
2. **Embedding:** NumPy vs Keras
3. **LSTM:** NumPy vs Keras
4. **Backward Pass:** Embeddings <-- LSTM <-- Dense
-----

## Implementation

* This implementation follows the keras implementation 1 (there are two), where the hidden state is multiplied with a recurrent weight kernel denoted by 'U'
* Implementation 2 concatenates the inputs, denoted by 'X', with the hidden state 'h', into term 'z', then multiplies this with the weight kernel. A lot of publications I've seen online utilize implementatin 2.

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

## **With Backward Pass**

* Not tested yet!
* Working on the Sequential class to propagate forward/backward 

```
from lstm import LSTM
from tokenizer import Vocabulary
from dense import Dense
from embedding import EmbeddingLayer
import numpy as np

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

e = EmbeddingLayer(vocab_size=v.size, hidden_dim=20)
batch1 = e.predict(X[0])

lstm = LSTM(units=100, features=20, seq_length=25)
init_state = {'h':np.zeros((100,)), 'c':np.zeros((100,))}
cache, state = lstm.forward(batch1, init_state)

dense = Dense(v.size)
final_out = dense.forward(state['h'])

init_state_grads = {'h':np.zeros_like(state['h']), 'c':np.zeros_like(state['c'])}

kernel_grads, recurrent_kernel_grads, state_grads, embedding_grads = lstm.backward(prediction=final_out,
                                                                                    actual=y[0],
                                                                                    state_gradients=init_state_grads, 
                                                                                    state=state, 
                                                                                    cache=cache,
                                                                                    dense_weights=dense.weights, 
                                                                                    first=True)
```
