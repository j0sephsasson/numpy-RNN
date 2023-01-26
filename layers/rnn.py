from typing import Any
import numpy as np


########## SINGLE LAYER RNN ##########
class RNNV1:
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

########## ARBITRARY LAYER RNN ##########
class RNNV2:
    def __init__(self, hidden_size, vocab_size, seq_length, num_layers):
        self.name = 'RNN'
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # model parameters
        self.Wxh = [np.random.randn(hidden_size, vocab_size)*0.01 for _ in range(num_layers)] # input to hidden
        self.Whh = [np.random.randn(hidden_size, hidden_size)*0.01 for _ in range(num_layers)] # hidden to hidden
        self.Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
        self.bh = [np.zeros((hidden_size, 1)) for _ in range(num_layers)] # hidden bias
        self.by = np.zeros((vocab_size, 1)) # output bias

        # memory variables for training (ada grad from karpathy's github)
        self.iteration, self.pointer = 0, 0
        self.mWxh = [np.zeros_like(w) for w in self.Wxh]
        self.mWhh = [np.zeros_like(w) for w in self.Whh] 
        self.mWhy = np.zeros_like(self.Why)
        self.mbh, self.mby = [np.zeros_like(b) for b in self.bh], np.zeros_like(self.by)
        self.loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

        self.running_loss = []

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
            hs[t] = np.copy(hprev)

            for l in range(self.num_layers):
                hs[t][l] = np.tanh(np.dot(self.Wxh[l], xs[t]) + np.dot(self.Whh[l], hs[t-1][l]) + self.bh[l]) # hidden state
            
            ys[t] = np.dot(self.Why, hs[t][-1]) + self.by # unnormalized log probabilities for next chars
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
            loss += -np.log(ps[t][y[t],0]) # softmax (cross-entropy loss)

        if kwds.get('loss_only', False):
            return loss
        return loss, hs[len(x)-1], {'xs':xs, 'hs':hs, 'ys':ys, 'ps':ps}

    def backward(self, targets, cache):
        """RNN Backward Pass"""

        xs, hs, ys, ps = cache['xs'], cache['hs'], cache['ys'], cache['ps']
        dWxh, dWhh, dWhy = [np.zeros_like(w) for w in self.Wxh], [np.zeros_like(w) for w in self.Whh], np.zeros_like(self.Why)
        dbh, dby = [np.zeros_like(b) for b in self.bh], np.zeros_like(self.by)
        dhnext = [np.zeros_like(h) for h in hs[0]]

        for t in reversed(range(len(xs))):

            dy = np.copy(ps[t])
            dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            dWhy += np.dot(dy, hs[t][-1].T)
            dby += dy

            for l in reversed(range(self.num_layers)):
                dh = np.dot(self.Why.T, dy) + dhnext[l]
                dhraw = (1 - hs[t][l] * hs[t][l]) * dh # backprop through tanh nonlinearity
                dbh[l] += dhraw
                dWxh[l] += np.dot(dhraw, xs[t].T)
                dWhh[l] += np.dot(dhraw, hs[t-1][l].T)
                dhnext[l] = np.dot(self.Whh[l].T, dhraw)

        return {'dWxh':dWxh, 'dWhh':dWhh, 'dWhy':dWhy, 'dbh':dbh, 'dby':dby}