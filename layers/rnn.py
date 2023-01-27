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
        self.seq_length = seq_length

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

            if kwds.get('dropout', False): # use dropout layer (mask)

                for l in range(self.num_layers):
                    dropout_mask = (np.random.rand(*hs[t-1][l].shape) < (1-0.5)).astype(float)
                    hs[t-1][l] *= dropout_mask
                    hs[t][l] = np.tanh(np.dot(self.Wxh[l], xs[t]) + np.dot(self.Whh[l], hs[t-1][l]) + self.bh[l]) # hidden state
                    hs[t][l] = hs[t][l] / (1 - 0.5)

            else: # no dropout layer (mask)

                for l in range(self.num_layers):
                    hs[t][l] = np.tanh(np.dot(self.Wxh[l], xs[t]) + np.dot(self.Whh[l], hs[t-1][l]) + self.bh[l]) # hidden state
            
                
            ys[t] = np.dot(self.Why, hs[t][-1]) + self.by # unnormalized log probabilities for next chars
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
            loss += -np.log(ps[t][y[t],0]) # softmax (cross-entropy loss)

        self.running_loss.append(loss)

        return loss, hs[len(x)-1], {'xs':xs, 'hs':hs, 'ps':ps}

    def backward(self, targets, cache):
        """RNN Backward Pass"""

        xs, hs, ps = cache['xs'], cache['hs'], cache['ps']
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

    def update(self, grads, lr):
        """Perform Parameter Update w/ Adagrad"""

        # unpack grads
        dWxh, dWhh, dWhy = grads['dWxh'], grads['dWhh'], grads['dWhy']
        dbh, dby = grads['dbh'], grads['dby']

        # loop through each layer
        for i in range(self.num_layers):
            # clip gradients to mitigate exploding gradients
            np.clip(dWxh[i], -5, 5, out=dWxh[i])
            np.clip(dWhh[i], -5, 5, out=dWhh[i])
            np.clip(dbh[i], -5, 5, out=dbh[i])

            # perform parameter update with Adagrad
            self.mWxh[i] += dWxh[i] * dWxh[i]
            self.Wxh[i] -= lr * dWxh[i] / np.sqrt(self.mWxh[i] + 1e-8)
            self.mWhh[i] += dWhh[i] * dWhh[i]
            self.Whh[i] -= lr * dWhh[i] / np.sqrt(self.mWhh[i] + 1e-8)
            self.mbh[i] += dbh[i] * dbh[i]
            self.bh[i] -= lr * dbh[i] / np.sqrt(self.mbh[i] + 1e-8)
        
        # clip gradients for Why and by
        np.clip(dWhy, -5, 5, out=dWhy)
        np.clip(dby, -5, 5, out=dby)

        # perform parameter update with Adagrad
        self.mWhy += dWhy * dWhy
        self.Why -= lr * dWhy / np.sqrt(self.mWhy + 1e-8)
        self.mby += dby * dby
        self.by -= lr * dby / np.sqrt(self.mby + 1e-8)

    def predict(self, hprev, seed_ix, n):
        """
        Make predictions using the trained RNN model.

        Parameters:
        hprev (numpy array): The previous hidden state.
        seed_ix (int): The seed letter index to start the prediction with.
        n (int): The number of characters to generate for the prediction.

        Returns:
        ixes (list): The list of predicted character indices.
        hs (numpy array): The final hidden state after making the predictions.
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []

        hs = {}
        hs[-1] = np.copy(hprev)

        for t in range(n):
            hs[t] = np.copy(hprev)
            
            for l in range(self.num_layers):
                hs[t][l] = np.tanh(np.dot(self.Wxh[l], x) + np.dot(self.Whh[l], hs[t-1][l]) + self.bh[l]) # hidden state
            
            ys = np.dot(self.Why, hs[t][-1]) + self.by # unnormalized log probabilities for next chars
            ps = np.exp(ys) / np.sum(np.exp(ys)) # probabilities for next chars
            
            ix = np.random.choice(range(self.vocab_size), p=ps.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)

        # return ixes , hs[n-1]
        return ixes

    def reset_params(self):
        # model parameters
        self.Wxh = [np.random.randn(self.hidden_size, self.vocab_size)*0.01 for _ in range(self.num_layers)] # input to hidden
        self.Whh = [np.random.randn(self.hidden_size, self.hidden_size)*0.01 for _ in range(self.num_layers)] # hidden to hidden
        self.Why = np.random.randn(self.vocab_size, self.hidden_size)*0.01 # hidden to output
        self.bh = [np.zeros((self.hidden_size, 1)) for _ in range(self.num_layers)] # hidden bias
        self.by = np.zeros((self.vocab_size, 1)) # output bias

        # memory variables for training (ada grad from karpathy's github)
        self.iteration, self.pointer = 0, 0
        self.mWxh = [np.zeros_like(w) for w in self.Wxh]
        self.mWhh = [np.zeros_like(w) for w in self.Whh] 
        self.mWhy = np.zeros_like(self.Why)
        self.mbh, self.mby = [np.zeros_like(b) for b in self.bh], np.zeros_like(self.by)
        self.loss = -np.log(1.0/self.vocab_size)*self.seq_length # loss at iteration 0

        self.running_loss = []