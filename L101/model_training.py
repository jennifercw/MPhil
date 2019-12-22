import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, LSTM, Dense, Attention, CuDNNLSTM, Embedding, Concatenate
from tensorflow.python.keras.callbacks import TensorBoard

from nltk import word_tokenize
from nltk.translate import bleu_score
import pickle

import numpy as np
import os


def delexicalise_both(mr_list, ref_list, delex_slots):
    slot_dict = {"name": "_NAME_", "eatType": "_EATTYPE_", "priceRange": "_PRICERANGE_",
                 "customer rating": "_CUSTOMERRATING_", "near": "_NEAR_", "food": "_FOOD_",
                 "area": "_AREA_"}
    assert(len(mr_list) == len(ref_list))
    new_mr_list = []
    new_ref_list = []
    for i in range(len(mr_list)):
        print(i)
        mr = mr_list[i]
        ref = ref_list[i]
        for slot in delex_slots:
            if mr.find(slot) == -1:
                continue
            start = mr.find(slot) + len(slot) + 1
            end = mr[start:].find(']') + start
            val = mr[start:end]
            if ref.find(val) == -1:
                continue
            mr = mr[:start] + slot_dict[slot] + mr[end:]
            while ref.find(val) != -1:
                start = ref.find(val)
                end = start + len(val)
                ref = ref[:start] + slot_dict[slot] + ref[end:]
        new_mr_list.append(mr)
        new_ref_list.append(ref)
    return new_mr_list, new_ref_list


def delexicalise_mrs(mr_list, delex_slots):
    slot_dict = {"name": "_NAME_", "eatType": "_EATTYPE_", "priceRange": "_PRICERANGE_",
                 "customer rating": "_CUSTOMERRATING_", "near": "_NEAR_", "food": "_FOOD_",
                 "area": "_AREA_"}
    new_mr_list = []
    for i in range(len(mr_list)):
        mr = mr_list[i]
        for slot in delex_slots:
            if mr.find(slot) == -1:
                continue
            start = mr.find(slot) + len(slot) + 1
            end = mr[start:].find(']') + start
            mr = mr[:start] + slot_dict[slot] + mr[end:]
        new_mr_list.append(mr)
    return new_mr_list


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
        mr_list, ref_list = delexicalise_both(mr_list, ref_list, delex_slots)
    else:
        mr_list = delexicalise_mrs(mr_list, delex_slots)
    assert(len(mr_list) == len(ref_list))
    char_set = pickle.load(open("char_set.obj", "rb"))
    char_set = sorted(list(char_set))
    num_chars = len(char_set)
    max_mr_length = max([len(mr) for mr in mr_list])
    max_ref_length = max([len(ref) for ref in ref_list])

    data_details = {'mr_input': mr_list, 'ref_sentences': ref_list, 'char_set': char_set, 'num_chars': num_chars,
                    'max_mr_length': max_mr_length, 'max_ref_length': max_ref_length}
    return data_details


def load_data_chars_reverse(filename="trainset.csv", delex_slots=(), delex_both=True):
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
    mr_list = [start + mr + end for mr in mr_list]
    if len(delex_slots) > 0:
        mr_list, ref_list = delexicalise_both(mr_list, ref_list, delex_slots)
    else:
        mr_list = delexicalise_mrs(mr_list, delex_slots)
    assert(len(mr_list) == len(ref_list))
    char_set = pickle.load(open("char_set.obj", "rb"))
    char_set = sorted(list(char_set))
    num_chars = len(char_set)
    max_mr_length = max([len(mr) for mr in mr_list])
    max_ref_length = max([len(ref) for ref in ref_list])

    data_details = {'mr_input': mr_list, 'ref_sentences': ref_list, 'char_set': char_set, 'num_chars': num_chars,
                    'max_mr_length': max_mr_length, 'max_ref_length': max_ref_length}
    return data_details


def load_data_words(filename="trainset.csv", delex_slots=(), delex_both=True):
    """
    Output is dictionary containing:
    "mr_input" : List of all MRs, "ref_list": List of all reference sentences
    "input_word_set" : List of all words in MRs, "output_word_set": all words in reference sentences
    "num_input_words" : number of words that appear in MRs, "num_output_words": number of words in ref sentences,
    "max_mr_length" : max length of MR seqs, "max_ref_length: max length of reference seqs
    """
    start = '\t'
    end = '\n'
    file = pd.read_csv(filename)
    mr_list = file['mr'].to_list()
    ref_list = file['ref'].to_list()
    ref_list = [start + ref + end for ref in ref_list]
    if len(delex_slots) > 0:
        # will need to add delex to word sets
        if delex_both:
            mr_list, ref_list = delexicalise_both(mr_list, ref_list, delex_slots)
        else:
            mr_list = delexicalise_mrs(mr_list, delex_slots)
    assert(len(mr_list) == len(ref_list))
    input_word_set = pickle.load(open("mr_word_set.obj", "rb"))
    output_word_set = pickle.load(open("ref_word_set.obj", "rb"))

    input_word_set = sorted(list(input_word_set))
    output_word_set = sorted(list(output_word_set))
    num_input_words = len(input_word_set)
    num_output_words = len(output_word_set)
    max_mr_length = max([len(word_tokenize(mr)) for mr in mr_list])
    max_ref_length = max([len(word_tokenize(ref)) for ref in ref_list])

    data_details = {'mr_input': mr_list, 'ref_sentences': ref_list, 'input_word_set': input_word_set,
                    'output_word_set' : output_word_set,'num_input_words': num_input_words,
                    'num_output_words': num_output_words, 'max_mr_length': max_mr_length,
                    'max_ref_length': max_ref_length}
    return data_details


def vectorise_chars(data, delex=False):
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


def vectorise_chars_reverse(data, delex=False):
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
    encoder_input = np.zeros((len(mr_list), max_ref_length, num_chars), dtype='float32')
    decoder_input = np.zeros((len(mr_list), max_mr_length, num_chars), dtype='float32')
    decoder_target = np.zeros((len(mr_list), max_mr_length, num_chars), dtype='float32')

    for i, (mr, ref) in enumerate(zip(mr_list, ref_list)):
        for j, c in enumerate(ref):
            encoder_input[i, j, char_index[c]] = 1.0
        for j, c in enumerate(mr):
            decoder_input[i, j, char_index[c]] = 1.0
            if j > 0:
                decoder_target[i, j - 1, char_index[c]] = 1.0

    data['char_index'] = char_index
    data['encoder_input'] = encoder_input
    data['decoder_input'] = decoder_input
    data['decoder_target'] = decoder_target
    return data


def vectorise_words(data, delex=False):
    """
    Output is data as before but additionally:
    "input_word_index" : word -> num index for word in MR
    "output_word_index" : word -> num index for word in ref sentence
    "encoder_input": matrix containing all of the encoder inputs. dimensions: number of texts * max length * number
    of tokens. One hot encoded
    "decoder_input" : same but for output sequences
    "decoder_target" : same but one step along
    """
    input_words = data['input_word_set']
    output_words = data['output_word_set']

    if not delex:
        mr_list = data['mr_input']
        ref_list = data['ref_sentences']
    else:
        mr_list = data["mr_list_delex"]
        ref_list = data["ref_list_delex"]
    num_input_words = data['num_input_words']
    num_output_words = data['num_output_words']
    max_mr_length = data['max_mr_length']
    max_ref_length = data['max_ref_length']

    input_word_index = dict([(w, i) for i, w in enumerate(input_words)])
    output_word_index = dict([(w, i) for i, w in enumerate(output_words)])
    encoder_input = np.zeros((len(mr_list), max_mr_length, num_input_words), dtype='float32')
    decoder_input = np.zeros((len(mr_list), max_ref_length), dtype='float32')
    decoder_target = np.zeros((len(mr_list), max_ref_length), dtype='float32')

    for i, (mr, ref) in enumerate(zip(mr_list, ref_list)):
        for j, w in enumerate(word_tokenize(mr)):
            encoder_input[i, j, input_word_index[w]] = 1.0
        for j, w in enumerate(word_tokenize(ref)):
            decoder_input[i, j] = output_word_index[w]
            if j > 0:
                decoder_target[i, j - 1] = output_word_index[w]

    data['input_word_index'] = input_word_index
    data['output_word_index'] = output_word_index
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
    attention = Attention(name='attention')
    atten_output = attention([decoder_outputs, encoder_outputs])
    atten_output.set_shape([None, None, lstm_units])
    concat = Concatenate()
    concat_output = concat([decoder_outputs, atten_output])
    decoder_dense = Dense(num_chars, input_shape=[None, None, lstm_units], activation='softmax',
                          name='softmax_output')
    decoder_output = decoder_dense(concat_output)

    model = Model([encoder_input, decoder_input], decoder_output)
    model.compile(optimizer=optimiser, loss='categorical_crossentropy')

    encoder_info = [state_h, state_c, encoder_outputs]
    encoder_model = Model(encoder_input, encoder_info)

    decoder_state_input_h = Input(shape=(lstm_units,))
    decoder_state_input_c = Input(shape=(lstm_units,))
    decoder_state_input_enc = Input(shape=[None, lstm_units])
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_input, initial_state=decoder_states_inputs)
    #dec_attention = Attention(name='attention')([decoder_outputs, decoder_state_input_enc])
    dec_attention = attention([decoder_outputs, decoder_state_input_enc])
    concat_output = concat([decoder_outputs, dec_attention])
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(concat_output)
    dec_state_inputs = [decoder_state_input_h, decoder_state_input_c, decoder_state_input_enc]
    decoder_model = Model([decoder_input] + dec_state_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model


def build_model_words(num_input_words, num_output_words, lstm_units=256, layers=1, optimiser='adam'):
    """
    encoder_input will be the large matrix with all the information
    Outputs from encoder LSTM are: encoder_outputs: hidden_state at every timestep; state_h: final hidden state;
    state_c: final cell state
    decoder_input will be matrix with each letter of output
    Initial state for decoder lstm is encoder states
    """
    encoder_input = Input(shape=(None, num_input_words), name='encoder_input')
    # Need to add embedding layer but it seems that that takes list of ints
    encoder_outputs, state_h, state_c = CuDNNLSTM(lstm_units, return_sequences=True, return_state=True,
                                                  name="encoder_lstm")(encoder_input)
    encoder_states = [state_h, state_c]

    decoder_input = Input(shape=(None, num_output_words), name='decoder_input')
    """input_embed_layer = Embedding(input_dim=num_output_words, output_dim=200)
    input_embed = input_embed_layer(decoder_input)
    input_embed.set_shape([None, None, None, None])"""
    decoder_lstm = CuDNNLSTM(lstm_units, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
    attention = Attention(name="attention")([decoder_outputs, encoder_outputs])
    attention.set_shape([None, None, lstm_units])
    decoder_dense = Dense(num_output_words, input_shape=[None, None, lstm_units], activation='softmax',
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
    #input_embed = input_embed_layer(decoder_input)
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_input, initial_state=decoder_states_inputs)
    dec_attention = Attention()([decoder_outputs, decoder_state_input_enc])
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(dec_attention)
    dec_state_inputs = [decoder_state_input_h, decoder_state_input_c, decoder_state_input_enc]
    decoder_model = Model([decoder_input] + dec_state_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model


def train_model_chars(model_name, delex_slots=(), lstm_units=256, batch_size=64, epochs=25, layers=1):
    training_data = load_data_chars(delex_slots=delex_slots)
    training_data = vectorise_chars(training_data)
    model, encoder_model, decoder_model = build_model_chars(training_data['num_chars'], lstm_units, layers=layers)
    model.fit(x=[training_data["encoder_input"], training_data["decoder_input"]], y=training_data["decoder_target"],
              batch_size=batch_size, epochs=epochs, validation_split=0.2)
    model.save(model_name + '.h5')
    encoder_model.save(model_name + '_encoder.h5')
    decoder_model.save(model_name + '_decoder.h5')


def train_model_words(model_name, delex_slots=(), lstm_units=256, batch_size=64, epochs=25, layers=1):
    training_data = load_data_words(delex_slots=delex_slots)
    training_data = vectorise_words(training_data)
    model, encoder_model, decoder_model = build_model_words(training_data["num_input_words"],
                                                            training_data["num_output_words"], lstm_units,
                                                            layers=layers)
    model.fit(x=[training_data["encoder_input"], training_data["decoder_input"]], y=training_data["decoder_target"],
              batch_size=batch_size, epochs=epochs, validation_split=0.2)


def train_reverse_model_chars(model_name, delex_slots=(), lstm_units=256, batch_size=64, epochs=25, layers=1):
    training_data = load_data_chars_reverse(delex_slots=delex_slots)
    training_data = vectorise_chars_reverse(training_data)
    model, encoder_model, decoder_model = build_model_chars(training_data['num_chars'], lstm_units, layers=layers)
    model.fit(x=[training_data["encoder_input"], training_data["decoder_input"]], y=training_data["decoder_target"],
              batch_size=batch_size, epochs=epochs, validation_split=0.2)
    model.save(model_name + 'reverse.h5')
    encoder_model.save(model_name + 'reverse_encoder.h5')
    decoder_model.save(model_name + 'reverse_decoder.h5')


#train_model_words("test")
train_model_chars("basic2", epochs=25)
#train_model_chars("delex_test", delex_slots=("name", "priceRange", "near", "area"), epochs=50)
#train_reverse_model_chars("reverse_test", epochs=15)
#train_model_chars("basic512", lstm_units=512)