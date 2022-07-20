import numpy as np
import string
import re


class Vocabulary:
    def __init__(self) -> None:
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.num_words = 0
        self.num_sentences = 0
        self.length_of_longest_sentence = 0

    def _add_word(self, word):
        if word not in self.word2index:
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
        for word in new.split(' '):
            self._add_word(word)
            
        if len(new.split(' ')) > self.length_of_longest_sentence:
            self.length_of_longest_sentence = len(new.split(' '))
      
        self.num_sentences += 1
        
    def pad_sequences(self, sequence):
        """
        Params:
        sequence --> numpy.array, Integer vector of tokenized words
        
        Returns:
        padded_sequence --> Integer vector of tokenized words with padding
        """
        return_arr = []
        
        for s in sequence:
            new = list(s)
            missing = self.length_of_longest_sentence - len(new)
            new.extend([0]*missing)
            return_arr.append(new)
            
        return np.vstack(return_arr)
    
    def compile_vocab(self, corpus):
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

    def _clean_sentence(self, sentence):
        new_string = re.sub(r'[^\w\s]', '', sentence)
        return new_string

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]
