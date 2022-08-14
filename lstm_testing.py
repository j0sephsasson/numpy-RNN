import pytest
import numpy as np
import tensorflow as tf
from layers.lstm import LSTM
from layers.embedding import EmbeddingLayer
from utils.tokenizer import Vocabulary

HIDDEN = 100
FEATURES = 20

BATCH_SIZE = 500
SEQ_LENGTH = 12

def get_lstm_shape_v1(lstm, data):
    """
    v1: return_sequences=FALSE
    """
    out = lstm.forward(data)

    return out.shape

def get_lstm_shape_v2(lstm, data):
    """
    v1: return_sequences=TRUE
    """
    out = lstm.forward(data, return_sequences=True)

    return out.shape


def test_answer():
    assert get_lstm_shape_v1(LSTM(units=HIDDEN, features=FEATURES), np.random.randn(BATCH_SIZE, SEQ_LENGTH, FEATURES)) == (BATCH_SIZE, HIDDEN)

def test_answer_two():
    assert get_lstm_shape_v2(LSTM(units=HIDDEN, features=FEATURES), np.random.randn(BATCH_SIZE, SEQ_LENGTH, FEATURES)) == (BATCH_SIZE, SEQ_LENGTH, HIDDEN)