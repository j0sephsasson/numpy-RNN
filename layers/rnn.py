from typing import Any
import numpy as np

class RNN:
    def __init__(self, hidden_size, vocab_size):
        self.name = 'RNN'
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # model parameters
        self.Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
        self.Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
        self.bh = np.zeros((hidden_size, 1)) # hidden bias
        self.by = np.zeros((vocab_size, 1)) # output bias

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """RNN Forward Pass"""

        x, y, hprev = kwds['inputs'], kwds['targets'], kwds['hprev']

        loss = 0
        xs, hs, ys, ps = {}, {}, {}, {} # inputs, hidden state, output, probabilities
        hs[-1] = np.copy(hprev)

        # forward pass
        for t in range(len(x)):
            xs[t] = np.zeros((self.vocab_size,1)) # encode in 1-of-k representation
            xs[t][x[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
            ys[t] = np.dot(self.Why, hs[t]) + self.by # unnormalized log probabilities for next chars
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
            loss += -np.log(ps[t][y[t],0]) # softmax (cross-entropy loss)

        return loss


if __name__ == "__main__":

    ## start with data
    data = open('/Users/joesasson/Desktop/open-source/numpy-RNN/data/input.txt', 'r').read() # should be simple plain text file

    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)

    char_to_idx = { ch:i for i,ch in enumerate(chars) }
    idx_to_char = { i:ch for i,ch in enumerate(chars) }

    ## hyper-params
    batch_size = 128
    seq_length = 8
    hidden_size = 256

    n, p = 0, 0

    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p+seq_length+1 >= len(data) or n == 0: 
        hprev = np.zeros((hidden_size,1)) # reset RNN memory
        p = 0 # go from start of data

    inputs_two = np.array([char_to_idx[ch] for ch in data[p:p+seq_length]])
    targets_two = np.array([char_to_idx[ch] for ch in data[p+1:p+seq_length+1]])

    rnn = RNN(hidden_size=hidden_size, vocab_size=vocab_size)
    print(rnn(inputs=inputs_two, targets=targets_two, hprev=hprev))