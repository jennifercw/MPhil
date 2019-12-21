import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, LSTM, Dense, Attention, CuDNNLSTM
from tensorflow.python.keras.callbacks import TensorBoard

from nltk.translate import bleu_score
import pickle

import numpy as np
import os


def load_data_chars(filename="trainset.csv", delex_slots=(), delex_both=True):
    """
    Output is dictionary containing:
    "mr_input" : List of all MRs, "ref_list": List of all reference sentences
    "char_set" : List of all characters in MRs and references
    "num_chars" : number of characters that appear
    "max_mr_length" : max length of MR seqs, "max_ref_length: max length of reference seqs
    """
    start = '\t'
    end = '\n'
    file = pd.read_csv(filename)
    mr_list = file['mr'].to_list()
    ref_list = file['ref'].to_list()
    ref_list = [start + ref + end for ref in ref_list]
    if len(delex_slots) > 0:
        mr_list_delex, ref_list_delex = delexicalise_both(mr_list, ref_list, delex_slots)
    else:
        mr_list_delex = delexicalise_mrs(mr_list, delex_slots)
    assert(len(mr_list) == len(ref_list))
    char_set = pickle.load(open("char_set.obj", "rb"))
    char_set = sorted(list(char_set))
    num_chars = len(char_set)
    max_mr_length = max([len(mr) for mr in mr_list])
    max_ref_length = max([len(ref) for ref in ref_list])

    data_details = {'mr_input': mr_list, 'ref_sentences': ref_list, 'char_set': char_set, 'num_chars': num_chars,
                    'max_mr_length': max_mr_length, 'max_ref_length': max_ref_length}
    return data_details


def vectorise_chars(data):
    """
    Output is data as before but additionally:
    "char_index" : char -> num index
    "encoder_input": matrix containing all of the encoder inputs. dimensions: number of texts * max length * number
    of tokens. One hot encoded
    "decoder_input" : same but for output sequences
    "decoder_target" : same but one step along
    """
    chars = data['char_set']
    mr_list = data['mr_input']
    ref_list = data['ref_sentences']
    num_chars = data['num_chars']
    max_mr_length = data['max_mr_length']
    max_ref_length = data['max_ref_length']

    char_index = dict([(c, i) for i, c in enumerate(chars)])
    encoder_input = np.zeros((len(mr_list), max_mr_length, num_chars), dtype='float32')
    decoder_input = np.zeros((len(mr_list), max_ref_length, num_chars), dtype='float32')
    decoder_target = np.zeros((len(mr_list), max_ref_length, num_chars), dtype='float32')

    for i, (mr, ref) in enumerate(zip(mr_list, ref_list)):
        for j, c in enumerate(mr):
            encoder_input[i, j, char_index[c]] = 1.0
        for j, c in enumerate(ref):
            decoder_input[i, j, char_index[c]] = 1.0
            if j > 0:
                decoder_target[i, j - 1, char_index[c]] = 1.0

    data['char_index'] = char_index
    data['encoder_input'] = encoder_input
    data['decoder_input'] = decoder_input
    data['decoder_target'] = decoder_target
    return data


def build_model_chars(num_chars, lstm_units=256, layers=1, optimiser='adam'):
    """
    encoder_input will be the large matrix with all the information
    Outputs from encoder LSTM are: encoder_outputs: hidden_state at every timestep; state_h: final hidden state;
    state_c: final cell state
    decoder_input will be matrix with each letter of output
    Initial state for decoder lstm is encoder states
    """
    encoder_input = Input(shape=(None, num_chars), name='encoder_input')
    encoder_outputs, state_h, state_c = CuDNNLSTM(lstm_units, return_sequences=True, return_state=True,
                                                  name="encoder_lstm")(encoder_input)
    encoder_states = [state_h, state_c]

    decoder_input = Input(shape=(None, num_chars), name='decoder_input')
    decoder_lstm = CuDNNLSTM(lstm_units, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
    attention = Attention()([decoder_outputs, encoder_outputs])
    attention.set_shape([None, None, lstm_units])
    decoder_dense = Dense(num_chars, input_shape=[None, None, lstm_units], activation='softmax',
                          name='softmax_output')
    decoder_output = decoder_dense(attention)

    model = Model([encoder_input, decoder_input], decoder_output)
    model.compile(optimizer=optimiser, loss='categorical_crossentropy')

    encoder_info = [state_h, state_c, encoder_outputs]
    encoder_model = Model(encoder_input, encoder_info)

    decoder_state_input_h = Input(shape=(lstm_units,))
    decoder_state_input_c = Input(shape=(lstm_units,))
    decoder_state_input_enc = Input(shape=[None, lstm_units])
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_input, initial_state=decoder_states_inputs)
    dec_attention = Attention()([decoder_outputs, decoder_state_input_enc])
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(dec_attention)
    dec_state_inputs = [decoder_state_input_h, decoder_state_input_c, decoder_state_input_enc]
    decoder_model = Model([decoder_input] + dec_state_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model


def train_model_chars(model_name, delex_slots=(), lstm_units=256, batch_size=32, epochs=25):
    training_data = load_data_chars(delex_slots=delex_slots)
    training_data = vectorise_chars(training_data)
    model, encoder_model, decoder_model = build_model_chars(training_data['num_chars'], lstm_units, layers=1)
    model.fit(x=[training_data["encoder_input"], training_data["decoder_input"]], y=training_data["decoder_target"],
              batch_size=batch_size, epochs=epochs, validation_split=0.2)
    model.save(model_name + '.h5')
    encoder_model.save(model_name + '_encoder.h5')
    decoder_model.save(model_name + '_decoder.h5')


train_model_chars("basic")