from typing import Any
import numpy as np

np.random.seed(99)

class RNN:
    def __init__(self, hidden_size, vocab_size, seq_length):
        self.name = 'RNN'
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # model parameters
        self.Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
        self.Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
        self.bh = np.zeros((hidden_size, 1)) # hidden bias
        self.by = np.zeros((vocab_size, 1)) # output bias

        # memory variables for training (ada grad from karpathy's github)
        self.iteration, self.pointer = 0, 0
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh) 
        self.mWhy = np.zeros_like(self.Why)
        self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(self.by)
        self.loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

        self.running_loss = []

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """RNN Forward Pass"""

        x, y, hprev = kwds['inputs'], kwds['targets'], kwds['hprev']

        lr = kwds['lr']

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

        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])

        for t in reversed(range(len(x))):
            dy = np.copy(ps[t])
            dy[y[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

        # perform parameter update with Adagrad
        for param, dparam, mem in zip([self.Wxh, self.Whh, 
                                        self.Why, self.bh, self.by], 
                                        [dWxh, dWhh, dWhy, dbh, dby], 
                                        [self.mWxh, self.mWhh, 
                                        self.mWhy, self.mbh, self.mby]):
            
            mem += dparam * dparam
            param += -lr * dparam / np.sqrt(mem + 1e-8) # adagrad update

        self.running_loss.append(loss)

        return loss, hs[len(x)-1]


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

    rnn = RNN(hidden_size=hidden_size, vocab_size=vocab_size, seq_length=seq_length)

    def train(rnn, epochs, data, lr=1e-1):

        for _ in range(epochs):

            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if rnn.pointer+seq_length+1 >= len(data) or rnn.iteration == 0: 
                hprev = np.zeros((hidden_size,1)) # reset RNN memory
                p = 0 # go from start of data

            x = [char_to_idx[ch] for ch in data[rnn.pointer:rnn.pointer+seq_length]]
            y = [char_to_idx[ch] for ch in data[rnn.pointer+1:rnn.pointer+seq_length+1]]

            # forward / backward pass single batch through network
            loss, hprev = rnn(inputs=x, targets=y, hprev=hprev, lr=lr)

            # update loss
            rnn.loss = rnn.loss * 0.999 + loss * 0.001

            ## show progress now and then
            if rnn.iteration % 1000 == 0: 
                print('iter {}, loss: {}'.format(rnn.iteration, rnn.loss))

            rnn.pointer += seq_length # move data pointer
            rnn.iteration += 1 # iteration counter 

            
    train(rnn=rnn, epochs=50000, data=data)