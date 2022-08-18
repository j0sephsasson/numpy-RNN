import numpy as np

class LSTM:
    def __init__(self, units, features, seq_length):
        """
        Initializes the LSTM layer
        
        Args:
            Units: int (num of LSTM units in layer)
            features: int (dimensionality of token embeddings)
            seq_length: int (num of timesteps per batch)
        """
        self.name = 'LSTM'
        self.hidden_dim = units
        self.dimensionality = features
        self.seq_length = seq_length

        self._init_params()
        
    def _init_orthogonal(self, param):
        """
        Initializes weight parameters orthogonally.

        Refer to this paper for an explanation of this initialization:
        https://arxiv.org/abs/1312.6120
        """
        if param.ndim < 2:
            raise ValueError("Only parameters with 2 or more dimensions are supported.")

        rows, cols = param.shape

        new_param = np.random.randn(rows, cols)

        if rows < cols:
            new_param = new_param.T

        # Compute QR factorization
        q, r = np.linalg.qr(new_param)

        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        d = np.diag(r, 0)
        ph = np.sign(d)
        q *= ph

        if rows < cols:
            q = q.T

        new_param = q

        return new_param
    
    def sigmoid(self, x, derivative=False):
        """
        Computes the element-wise sigmoid activation function for an array x.

        Args:
         `x`: the array where the function is applied
         `derivative`: if set to True will return the derivative instead of the forward pass
        """
        x_safe = x + 1e-12
        f = 1 / (1 + np.exp(-x_safe))

        if derivative: # Return the derivative of the function evaluated at x
            return f * (1 - f)
        else: # Return the forward pass of the function at x
            return f
    
    def tanh(self, x, derivative=False):
        """
        Computes the element-wise tanh activation function for an array x.

        Args:
         `x`: the array where the function is applied
         `derivative`: if set to True will return the derivative instead of the forward pass
        """
        x_safe = x + 1e-12
        f = (np.exp(x_safe)-np.exp(-x_safe))/(np.exp(x_safe)+np.exp(-x_safe))

        if derivative: # Return the derivative of the function evaluated at x
            return 1-f**2
        else: # Return the forward pass of the function at x
            return f
    
    def _init_params(self):
        """
        Initializes the weight and biases of the layer
        
            -- Initialize weights according to https://arxiv.org/abs/1312.6120 (_init_orthogonal)
            -- Initialize weights according to https://github.com/keras-team/keras/blob/master/keras/layers/rnn/lstm.py
            -- Assumptions: Batch_First=True (PyTorch) or time_major=False (keras)
        """
        self.kernel = self._init_orthogonal(np.random.randn(self.dimensionality, self.hidden_dim * 4))
        self.recurrent_kernel = self._init_orthogonal(np.random.randn(self.hidden_dim, self.hidden_dim * 4))
        self.bias = np.random.randn(self.hidden_dim * 4, )

        self.kernel_i = self.kernel[:, :self.hidden_dim]
        self.kernel_f = self.kernel[:, self.hidden_dim: self.hidden_dim * 2]
        self.kernel_c = self.kernel[:, self.hidden_dim * 2: self.hidden_dim * 3]
        self.kernel_o = self.kernel[:, self.hidden_dim * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.hidden_dim]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.hidden_dim: self.hidden_dim * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.hidden_dim * 2: self.hidden_dim * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.hidden_dim * 3:]

        self.bias_i = self.bias[:self.hidden_dim]
        self.bias_f = self.bias[self.hidden_dim: self.hidden_dim * 2]
        self.bias_c = self.bias[self.hidden_dim * 2: self.hidden_dim * 3]
        self.bias_o = self.bias[self.hidden_dim * 3:]

    def forward(self, inputs, state):
        
        inputs_i = inputs
        inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs
       
        h_tm1_i = state['h']
        h_tm1_f = state['h']
        h_tm1_c = state['h']
        h_tm1_o = state['h']

        x_i = np.dot(inputs_i, self.kernel_i) + self.bias_i
        x_f = np.dot(inputs_f, self.kernel_f) + self.bias_f
        x_c = np.dot(inputs_c, self.kernel_c) + self.bias_c
        x_o = np.dot(inputs_o, self.kernel_o) + self.bias_o

        f = self.sigmoid(x_f + np.dot(h_tm1_f, self.recurrent_kernel_f))
        i = self.sigmoid(x_i + np.dot(h_tm1_i, self.recurrent_kernel_i))
        o = self.sigmoid(x_o + np.dot(h_tm1_o, self.recurrent_kernel_o))
        cbar = self.sigmoid(x_c + np.dot(h_tm1_c, self.recurrent_kernel_c))
        
        c = (f * state['c']) + (i * cbar)
            
        ht = o * self.tanh(c)
        
        cache = {'i':i, 'f':f, 'cbar':cbar, 'o':o, 'inputs':inputs}
        state = {'h':ht, 'c':c}

        return cache, state
        
    def backward(self, prediction, actual, state, cache, dense_weights):
        dscores = np.copy(prediction)
        dscores[range(self.seq_length), actual] -= 1

        i, f, cbar, o = cache['i'], cache['f'], cache['cbar'], cache['o']
        h, c = state['h'], state['c']

        # Hidden to output (dense) gradient
        dWy = np.dot(h.T, dscores)
        dh = np.dot(dscores, dense_weights.T)
        dby = np.sum(dscores, axis=0, keepdims=True)
        dby = dby.reshape(dby.shape[1],)
        
        # Gradient for o
        do = self.tanh(c) * dh
        do = self.sigmoid(o, derivative=True) * do

        # Gradient for cbar
        dcbar = o * dh * self.tanh(c, derivative=True)
            
        # Gradient for f
        df = cbar * dcbar
        df = self.sigmoid(f, derivative=True) * df
        
        # Gradient for i
        di = c * dcbar
        di = self.sigmoid(i, derivative=True) * di
        
        # Gradient for c
        dc = i * dcbar
        dc = self.tanh(c, derivative=True) * dc
        
        # Gate gradients, just a normal fully connected layer gradient
        # We backprop into the kernel, recurrent_kernel, bias, inputs (embedding), & hidden state
        dWf = np.dot(cache['inputs'].T, df) # --> kernel
        dXf = np.dot(df, self.kernel_f.T) # --> embedding
        dUf = np.dot(h.T, df) # --> recurrent kernel
        dhf = np.dot(df, self.recurrent_kernel_f) # --> hidden state
        dbf = np.sum(df, axis=0, keepdims=True) # --> bias
        dbf = dbf.reshape(dbf.shape[1],)

        dWi = np.dot(cache['inputs'].T, di)
        dXi = np.dot(di, self.kernel_i.T)
        dUi = np.dot(h.T, di)
        dhi = np.dot(di, self.recurrent_kernel_i)
        dbi = np.sum(di, axis=0, keepdims=True)
        dbi = dbi.reshape(dbi.shape[1],)
        
        dWo = np.dot(cache['inputs'].T, do)
        dXo = np.dot(do, self.kernel_o.T)
        dUo = np.dot(h.T, do)
        dho = np.dot(do, self.recurrent_kernel_o)
        dbo = np.sum(do, axis=0, keepdims=True)
        dbo = dbo.reshape(dbo.shape[1],)
        
        dWc = np.dot(cache['inputs'].T, dc)
        dXc = np.dot(dc, self.kernel_c.T)
        dUc = np.dot(h.T, dc)
        dhc = np.dot(dc, self.recurrent_kernel_c)
        dbc = np.sum(dc, axis=0, keepdims=True)
        dbc = dbc.reshape(dbc.shape[1],)
        
        # As X was used in multiple gates, the gradient must be accumulated here
        dX = dXo + dXc + dXi + dXf

        kernel_grads = dict(Wf=dWf, Wi=dWi, Wc=dWc, Wo=dWo, Wy=dWy, bf=dbf, bi=dbi, bc=dbc, bo=dbo, by=dby)
        recurrent_kernel_grads = dict(Uf=dUf, Ui=dUi, Uc=dUc, Uo=dUo)
        embedding_grads = dict(dX=dX)
        
        return kernel_grads, recurrent_kernel_grads, embedding_grads