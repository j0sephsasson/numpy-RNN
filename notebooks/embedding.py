import numpy as np

class EmbeddingLayer:
    def __init__(self, vocab_size, hidden_dim):
        self.name = 'Embedding'
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.weights = np.random.randn(vocab_size, hidden_dim) ## (vocab_size, hidden_dim)

    def forward(self, array):
        """
        PARAMS:
          array: 
           -- integer matrix of batch_size x seq_length
           or
           -- integer matrix of seq_length x 1 for a single batch

        RETURNS:
          array:
           -- integer matrix of batch_size x seq_length x hidden_dim
           or
           -- seq_length x hidden_dim for a single batch
        """
        assert np.max(array) <= self.vocab_size

        return np.array([self.weights[i] for i in array])   