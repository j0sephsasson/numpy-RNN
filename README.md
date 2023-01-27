# numpy-RNN
**Vanilla-RNN & LSTM-RNN Implementation w/ NumPy**


## Vanilla RNN Usage
#### Step 1: Data
* > data = open('path-to-data', 'r').read() # should be simple plain text file
* > chars = list(set(data)) <br> <br> data_size, vocab_size = len(data), len(chars) 
* > char_to_idx = { ch:i for i,ch in enumerate(chars) } <br> <br> idx_to_char = { i:ch for i,ch in enumerate(chars) }


#### Step 2: Initialize RNN
``` 
    ## Initialize RNN ##
    num_layers = 2
    hidden_size = 128
    seq_length = 12

    rnn = RNN(hidden_size=hidden_size, vocab_size=vocab_size, seq_length=seq_length, num_layers=num_layers)
```

#### Step 3: Train RNN

```
    def train(rnn, epochs, data, lr=1e-1):

        for _ in range(epochs):

            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if rnn.pointer+seq_length+1 >= len(data) or rnn.iteration == 0:

                if rnn.num_layers == 1:
                    hprev = np.zeros((hidden_size,1)) # reset RNN memory
                else:
                    hprev = [np.zeros((hidden_size, 1)) for _ in range(rnn.num_layers)]  # reset RNN memory

                rnn.pointer = 0 # go from start of data

            x = [char_to_idx[ch] for ch in data[rnn.pointer:rnn.pointer+seq_length]]
            y = [char_to_idx[ch] for ch in data[rnn.pointer+1:rnn.pointer+seq_length+1]]

            if rnn.num_layers == 1:
                # forward / backward pass single batch through network
                loss, hprev = rnn(inputs=x, targets=y, hprev=hprev, lr=lr)
            else:
                loss, hprev, cache = rnn(inputs=x, targets=y, hprev=hprev)
                grads = rnn.backward(targets=y, cache=cache)
                rnn.update(grads=grads, lr=1e-1)

            # update loss
            rnn.loss = rnn.loss * 0.999 + loss * 0.001

            ## show progress now and then
            if rnn.iteration % 1000 == 0: 
                print('iter {}, loss: {}'.format(rnn.iteration, rnn.loss))

                sample_ix = rnn.predict(hprev, x[0], 200)
                txt = ''.join(idx_to_char[ix] for ix in sample_ix)
                print('Sample')
                print ('----\n {} \n----'.format(txt))

            rnn.pointer += seq_length # move data pointer
            rnn.iteration += 1 # iteration counter 

train(rnn=rnn, epochs=50000, data=data)
```
**Metrics** 

![Loss](https://github.com/j0sephsasson/numpy-rnn/blob/main/loss.png?raw=true)

<br>
<br>

## Notes About Implementation
#### NumPy vs PyTorch 
* Obviously took a lot of inspiration from (https://gist.github.com/karpathy/d4dee566867f8291f086), shoutout to that legend.
* The main difference is that in the NumPy RNN, we are processing the input sequences one at a time, which is also known as online or sequential learning. In contrast, in PyTorch, the input is passed as a 3D tensor where the first dimension corresponds to the batch size, the second dimension corresponds to the sequence length, and the third dimension corresponds to the hidden size. This approach is also known as batch learning or offline learning.
* The NumPy Vanilla RNN is a 'many-to-many' network, meaning it takes sequences as input and returns sequences as output

<br>
<br>

## Next Steps
* Create a simple tokenizer module [ ]
* Implement a dropout layer (mask) [X]
* Abstract it even more to allow different hidden sizes in different RNN layers [ ]
* Create an Embedding layer & implement, (will need to change from sequential learning to batch learning) [ ]