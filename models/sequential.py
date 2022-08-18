import numpy as np

class LSTMSequential:
    def __init__(self):
        self.network = {}
        self.losses = []
        
    def add(self, layer):
        self.network[layer.name] = layer
        
    def _init_hidden(self):
        hidden = self.network['LSTM'].hidden_dim
        
        state = {'h':np.zeros((hidden,)), 'c':np.zeros((hidden,))}
    
        return state
    
    def _calculate_loss(self, predictions, actual):
        samples = self.network['LSTM'].seq_length ## t_steps

        correct_logprobs = -np.log(predictions[range(samples),actual])
        data_loss = np.sum(correct_logprobs)/samples

        return data_loss
    
    def _step(self, kernel_grads, recurrent_grads, embedding_grads, cache, lr):
        self.network['Dense'].weights -= lr * kernel_grads['Wy']
        self.network['Dense'].bias -= lr * kernel_grads['by']
        
        self.network['Embedding'].weights[cache['embedding_inputs']] -= lr * embedding_grads['dX']
            
        self.network['LSTM'].kernel_f -= lr * kernel_grads['Wf']
        self.network['LSTM'].kernel_i -= lr * kernel_grads['Wi']
        self.network['LSTM'].kernel_c -= lr * kernel_grads['Wc']
        self.network['LSTM'].kernel_o -= lr * kernel_grads['Wo']

        self.network['LSTM'].recurrent_kernel_f -= lr * recurrent_grads['Uf']
        self.network['LSTM'].recurrent_kernel_i -= lr * recurrent_grads['Ui']
        self.network['LSTM'].recurrent_kernel_c -= lr * recurrent_grads['Uc']
        self.network['LSTM'].recurrent_kernel_o -= lr * recurrent_grads['Uo']
        
        self.network['LSTM'].bias_f -= lr * kernel_grads['bf']
        self.network['LSTM'].bias_i -= lr * kernel_grads['bi']
        self.network['LSTM'].bias_c -= lr * kernel_grads['bc']
        self.network['LSTM'].bias_o -= lr * kernel_grads['bo']
    
    def _train_step(self, X_train, y_train, state, lr):
        assert('Embedding' in self.network and 'LSTM' in self.network and 'Dense' in self.network)
        
        loss = 0
        
        for idx in range(0, X_train.shape[0]):
        
            lstm_inp = self.network['Embedding'].forward(X_train[idx])
            cache, state = self.network['LSTM'].forward(lstm_inp, state)
            final_out = self.network['Dense'].forward(state['h'])
            
            cache['embedding_inputs'] = np.copy(X_train[idx])
            
            l = self._calculate_loss(predictions=final_out, actual=y_train[idx])
            loss+=l
                
            kernel_grads, r_kernel_grads, embed_grads = \
                                        self.network['LSTM'].backward(prediction=final_out, 
                                        actual=y_train[idx], 
                                        state=state,
                                        cache=cache,
                                        dense_weights=self.network['Dense'].weights)
            
            self._step(kernel_grads=kernel_grads, recurrent_grads=r_kernel_grads, 
                       embedding_grads=embed_grads, cache=cache, lr=lr)
            
        self.losses.append(loss/X_train.shape[0])
            
        return loss/X_train.shape[0], state

            
    def train(self, X_train, y_train, epochs, lr=0.01):
        init_state = self._init_hidden()

        for e in range(epochs):
            
            if e == 0:
                loss, state = self._train_step(X_train=X_train, y_train=y_train, state=init_state, lr=lr)
            else:
                loss, state = self._train_step(X_train=X_train, y_train=y_train, state=state, lr=lr)

            print('LOSS: {}, EPOCH: {}'.format(loss, str(e+1)))