import numpy as np

class EmbeddingLayer:
    def __init__(self, vocab_size, hidden_dim):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.weights = np.random.randn(vocab_size, hidden_dim) ## (vocab_size, hidden_dim)

    def predict(self, array):
        """
        PARAMS:
          array: 
           -- integer matrix of batch_size x seq_length

        RETURNS:
          array:
           -- integer matrix of batch_size x seq_length x hidden_dim
           -- the word vectors for each word in the tokenized input
        """
        assert np.max(array) <= self.vocab_size

        return np.array([self.weights[i] for i in array])    