import numpy as np

class LSTM:
    def __init__(self, units, features):
        """
        Initializes the LSTM layer
        
        Args:
            Units: int (num of LSTM units in layer)
            features: int (dimensionality of token embeddings)
        """
        self.hidden_dim = units
        self.dimensionality = features
        
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
    
    def softmax(self, x):
        """
        Computes the softmax for an array x.

        Args:
         `x`: the array where the function is applied
         `derivative`: if set to True will return the derivative instead of the forward pass
        """
        x_safe = x + 1e-12
        f = np.exp(x_safe) / np.sum(np.exp(x_safe))

        # Return the forward pass of the function at x
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

    def forward(self, inputs, return_sequences=False):
        """
        Performs one full forward pass through the layer

        Args:
            inputs: 3D array of shape (batch_size, seq_length, dimensionality)
            return_sequences: return the full sequence of hidden states or just the last one (per batch)
        """

        self._init_params()

        h_tm1 = np.zeros((self.hidden_dim,))
        c_tm1 = np.zeros((self.hidden_dim,))
        
        self.h_state_out = []
        
        for batch in inputs:
        
            inputs_i = batch
            inputs_f = batch
            inputs_c = batch
            inputs_o = batch

            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1

            x_i = np.dot(inputs_i, self.kernel_i) + self.bias_i
            x_f = np.dot(inputs_f, self.kernel_f) + self.bias_f
            x_c = np.dot(inputs_c, self.kernel_c) + self.bias_c
            x_o = np.dot(inputs_o, self.kernel_o) + self.bias_o

            f = self.sigmoid(x_f + np.dot(h_tm1_f, self.recurrent_kernel_f))
            i = self.sigmoid(x_i + np.dot(h_tm1_i, self.recurrent_kernel_i))
            o = self.sigmoid(x_o + np.dot(h_tm1_o, self.recurrent_kernel_o))
            cbar = self.sigmoid(x_c + np.dot(h_tm1_c, self.recurrent_kernel_c))
            c = (f * c_tm1) + (i * cbar)
            ht = o * self.tanh(c)
            
            if return_sequences == True:
                self.h_state_out.append(ht)
            else:
                self.h_state_out.append(ht[-1])
            
            h_tm1 = ht
            c_tm1 = c
        
        return np.array(self.h_state_out)