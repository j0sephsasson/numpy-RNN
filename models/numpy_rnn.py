import numpy as np

import sys
sys.path.append('../')

from layers.rnn import RNNV1, RNNV2

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


if __name__ == "__main__":

    ## start with data
    data = open('/Users/joesasson/Desktop/open-source/numpy-RNN/data/input.txt', 'r').read() # should be simple plain text file

    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)

    char_to_idx = { ch:i for i,ch in enumerate(chars) }
    idx_to_char = { i:ch for i,ch in enumerate(chars) }

    ## hyper-params
    seq_length = 8
    hidden_size = 100

    rnnv1 = RNNV1(hidden_size=hidden_size, vocab_size=vocab_size, seq_length=seq_length)
    rnnv2 = RNNV2(hidden_size=hidden_size, vocab_size=vocab_size, seq_length=seq_length, num_layers=2)
    
    train(rnn=rnnv2, epochs=15000, data=data)