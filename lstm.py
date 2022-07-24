import numpy as np

class LSTM:
    def __init__(self, units, seq_length, vocab_size, features):
        """
        Initializes the LSTM layer
        
        Args:
            Units: int (num of LSTM units in layer)
            features: int (dimensionality of token embeddings)
            seq_length: int (num of tokens at each timestep)
            vocab_size: int (num of unique tokens in vocab)
        """
        self.hidden_dim = units
        self.dimensionality = features
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        # Initialize hidden state as zeros
        self.h = np.zeros((units, features))
        self.c = np.zeros((units, features))
        
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
        
        Initialize weights according to https://arxiv.org/abs/1312.6120 (_init_orthogonal)
        """
        
        # Weight matrix (forget gate)
        self.W_f = self._init_orthogonal(np.random.randn(self.hidden_dim , self.seq_length))

        # Bias for forget gate
        self.b_f = np.zeros((self.hidden_dim , 1))

        # Weight matrix (input gate)
        self.W_i = self._init_orthogonal(np.random.randn(self.hidden_dim , self.seq_length))

        # Bias for input gate
        self.b_i = np.zeros((self.hidden_dim , 1))

        # Weight matrix (candidate)
        self.W_g = self._init_orthogonal(np.random.randn(self.hidden_dim , self.seq_length))

        # Bias for candidate
        self.b_g = np.zeros((self.hidden_dim , 1))

        # Weight matrix of the output gate
        self.W_o = self._init_orthogonal(np.random.randn(self.hidden_dim , self.seq_length))
        self.b_o = np.zeros((self.hidden_dim , 1))

        # Weight matrix relating the hidden-state to the output
        self.W_v = self._init_orthogonal(np.random.randn(self.vocab_size, self.hidden_dim))
        self.b_v = np.zeros((self.vocab_size, 1))