import numpy as np
import re


class Vocabulary:
    def __init__(self) -> None:
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.sentences = []
        self.tokens = []
        self.num_words = 0
        self.num_sentences = 0

    def _add_word(self, word):
        if word not in self.word2index:
            self.tokens.append(word)
            self.word2count[word] = 1
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def _add_sentence(self, sentence):
        sentence = sentence.lower()
        new = self._clean_sentence(sentence=sentence)
        new = new.replace('\n', '')
        self.sentences.append(new)
        
        for word in new.split(' '):
            if word != '':
                self._add_word(word)
            else:
                continue
      
        self.num_sentences += 1
        
    def pad_sequences(self, sequence, length=None):
        """
        Default: Pad an input sequence to be the same as self.seq_length
        
        Alternative: Pad an input sequence to the 'length' param
        
        Keras: Pads input sequences with length of longest sequence
        
        Params:
        sequence --> np.array[numpy.array], integer matrix of tokenized words
        
        Returns:
        padded_sequence --> np.array[numpy.array], integer matrix of tokenized words with padding
        """
        return_arr = []
        
        for s in sequence:
            new = list(s)
            
            if not length:
                missing = self.seq_length - len(new)
            else:
                missing = length - len(new)
                
            new.extend([0]*missing)
            return_arr.append(new)
            
        return np.vstack(return_arr)
    
    def _sort_by_frequency(self):
        sorted_count = dict(sorted(self.word2count.items(), key=lambda x:x[1], reverse=True))

        self.word2index = {}
        
        count = 0 ## start at 1 to copy keras --> 0 is reserved for padding (this is how keras does it)
        for k,v in sorted_count.items():
            self.word2index[k] = count
            count += 1
        
        self.index2word = {v:k for k,v in self.word2index.items()}
        
        return self
    
    def _compile_vocab(self, corpus):
        """
        Creates vocabulary

        Params:
        Corpus --> List[str]
        
        Returns:
        self
        """
        for s in corpus:
            self._add_sentence(s)

        assert len(self.word2count) == len(self.word2index) == len(self.index2word)
        self.size = len(self.word2count)
        
        self._sort_by_frequency()
        
    def tokenize(self, corpus, seq_length):
        """
        Creates sequences of tokens

        Params:
        Corpus --> List[str]
        
        Returns:
        Token Sequences --> List[str]
        """
        self._compile_vocab(corpus)
        self.seq_length = seq_length
        self.token_sequences = []
        
        for i in range(seq_length, self.size):
            seq = self.tokens[i-seq_length:i]
            seq = [self.word2index[i] for i in seq]
            self.token_sequences.append(seq)
        
        return np.array(self.token_sequences)

    def _clean_sentence(self, sentence):
        new_string = re.sub(r'[^\w\s]', '', sentence)
        return new_string

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]