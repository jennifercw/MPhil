import tensorflow as tf
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Attention
import numpy as np
import pandas as pd
import math
import random
import pickle
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score


def delexicalise_both(mr_list, ref_list, delex_slots):
    slot_dict = {"name": "_NAME_", "eatType": "_EATTYPE_", "priceRange": "_PRICERANGE_",
                 "customer rating": "_CUSTOMERRATING_", "near": "_NEAR_", "food": "_FOOD_",
                 "area": "_AREA_"}
    assert(len(mr_list) == len(ref_list))
    new_mr_list = []
    new_ref_list = []
    for i in range(len(mr_list)):
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


def relexicalise_sentence(output_sent, original_mr, delex_slots):
    slot_dict = {"name": "_NAME_", "eatType": "_EATTYPE_", "priceRange": "_PRICERANGE_",
                 "customer rating": "_CUSTOMERRATING_", "near": "_NEAR_", "food": "_FOOD_",
                 "area": "_AREA_"}
    for slot in delex_slots:
        if original_mr.find(slot) == -1:
            continue
        start = original_mr.find(slot) + len(slot) + 1
        end = original_mr[start:].find(']') + start
        val = original_mr[start:end]
        while output_sent.find(slot_dict[slot]) != -1:
            start = output_sent.find(slot_dict[slot])
            end = start + len(slot_dict[slot])
            output_sent = output_sent[:start] + val + output_sent[end:]
    return output_sent


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
    # Devset seem to be grouped so shuffle these
    # Should probably save a shuffled version so we can compare
    mr_list = file['mr'].to_list()
    ref_list = file['ref'].to_list()
    pairs = [(mr, ref) for mr, ref in zip(mr_list, ref_list)]
    random.shuffle(pairs)
    mr_list = [p[0] for p in pairs]
    ref_list = [p[1] for p in pairs]
    ref_list = [start + ref + end for ref in ref_list]
    mr_list_delex = []
    ref_list_delex = []
    if len(delex_slots) > 0:
        if delex_both:
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
    if len(mr_list_delex) > 0:
        data_details['mr_list_delex'] = mr_list_delex
    if len(ref_list_delex) > 0:
        data_details['ref_list_delex'] = ref_list_delex
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
    if not delex:
        mr_list = data['mr_input']
        ref_list = data['ref_sentences']
    else:
        mr_list = data["mr_list_delex"]
        ref_list = data["ref_list_delex"]
    num_chars = data['num_chars']
    max_mr_length = data['max_mr_length']
    max_ref_length = data['max_ref_length']

    char_index = dict([(c, i) for i, c in enumerate(chars)])
    encoder_input = np.zeros((len(mr_list), max_mr_length, num_chars), dtype='float32')
    decoder_input = np.zeros((len(mr_list), max_ref_length, num_chars), dtype='float32')
    decoder_target = np.zeros((len(mr_list), max_ref_length, num_chars), dtype='float32')

    for i, (mr, ref) in enumerate(zip(mr_list, ref_list)):
        for j, c in enumerate(mr):
            encoder_input[i, j, char_index[c]] = 1.
        for j, c in enumerate(ref):
            decoder_input[i, j, char_index[c]] = 1.
            if j > 0:
                decoder_target[i, j - 1, char_index[c]] = 1.

    data['char_index'] = char_index
    data['encoder_input'] = encoder_input
    data['decoder_input'] = decoder_input
    data['decoder_target'] = decoder_target
    return data


def load_models(model_name):
    model = load_model(model_name + '.h5')
    encoder = load_model(model_name + '_encoder.h5')
    decoder = load_model(model_name + '_decoder.h5')
    return [model, encoder, decoder]


def create_reverse_indices(data):
    data['reverse_char_index'] = dict((i, char) for char, i in data["char_index"].items())
    return data


def decode_beam_search(input, data, encoder, decoder, alpha=-1.0, beam_width=10, penalty_dict=None):
    states_value = encoder.predict(input)
    target = np.zeros((1, 1, data['num_chars']))
    target[0, 0, data['char_index']['\t']] = 1.
    input_val = [target] + states_value
    output_tokens, h, c, attention = decoder.predict(input_val)
    sampled_token_indices = output_tokens[0, -1, :].argsort()[-beam_width:][::-1]
    probs = [math.log(output_tokens[0, -1, i], 2) for i in sampled_token_indices]
    sampled_chars = [data["reverse_char_index"][sampled_token_index] for sampled_token_index in sampled_token_indices]
    top_paths = []

    for i, pred in enumerate(sampled_token_indices):
        decoded = sampled_chars[i]
        prob = probs[i]
        target = np.zeros((1, 1, data['num_chars']))
        target[0, 0, sampled_token_indices[i]] = 1.
        states_value = [h, c, states_value[2]]
        top_paths.append([target, prob, decoded, False, states_value])
    top_paths.sort(key=lambda x: x[1], reverse=True)
    stop_condition = False

    while not stop_condition:
        all_paths = []
        for path in top_paths:
            if path[3]:
                all_paths.append(path)
                continue
            target = path[0]
            prob = path[1]
            input_val = [target] + path[4]
            output_tokens, h, c, attention = decoder.predict(input_val)
            sampled_token_indices = output_tokens[0, -1, :].argsort()[-beam_width:][::-1]
            probs = [math.log(output_tokens[0, -1, i], 2) for i in sampled_token_indices]
            sampled_chars = [data["reverse_char_index"][sampled_token_index]
                             for sampled_token_index in sampled_token_indices]
            for i, pred in enumerate(sampled_token_indices):
                decoded = path[2] + sampled_chars[i]
                prob += probs[i]
                target = np.zeros((1, 1, data['num_chars']))
                target[0, 0, sampled_token_indices[i]] = 1.
                states_value = [h, c, path[4][2]]
                sent_stop = False
                if sampled_chars[i] == '\n' or len(decoded) > data['max_ref_length']:
                    sent_stop = True
                all_paths.append([target, prob, decoded, sent_stop, states_value])
        all_paths.sort(key=lambda x: x[1], reverse=True)
        top_paths = all_paths[:beam_width]
        stop_condition = True
        for path in top_paths:
            if path[3] is False:
                stop_condition = False

    for path in top_paths:
        if alpha == -1.0:
            lp = len(path[2])
        else:
            lp = (math.pow(5 + len(path[2]), alpha) / math.pow(5 + 1, alpha))
        path[1] = path[1] / lp
    top_paths.sort(key=lambda x: x[1], reverse=True)
    return [p[2] for p in top_paths]


def test_model_chars(model_name, num_examples=200, delex_slots=()):
    # https://stackoverflow.com/questions/53867351/how-to-visualize-attention-weights
    data = load_data_chars("devset.csv", delex_slots=delex_slots, delex_both=False)
    data = vectorise_chars(data)
    data = create_reverse_indices(data)
    model, encoder, decoder = load_models(model_name)
    decoder = Model(inputs=decoder.input, outputs=[decoder.output, decoder.get_layer('attention').output])
    bleus = []
    max_bleus = []
    meteors = []

    for seq_index in range(num_examples):
        print("Test example: " + str(seq_index) + "/" + str(len(data["encoder_input"])))

        input = data["encoder_input"][seq_index: seq_index + 1]
        correct = data['ref_sentences'][seq_index].strip("\t\n")
        paths = decode_beam_search(input, data, encoder, decoder)

        print('-')
        print(paths)
        if len(delex_slots) > 0:
            for i in range(len(paths)):
                paths[i] = relexicalise_sentence(paths[i], data['mr_input'][seq_index], delex_slots)
        decoded_sentence = paths[0]
        max_bleu = max(sentence_bleu(references=[correct], hypothesis=p) for p in paths)

        print('Input sentence:', data['mr_input'][seq_index])
        print('Decoded sentence:', decoded_sentence)
        print("Correct: ", correct)

        bleu = sentence_bleu(references=[correct], hypothesis=decoded_sentence)
        bleus.append(bleu)
        max_bleus.append(max_bleu)

        meteor = meteor_score(references=[correct], hypothesis=decoded_sentence)
        meteors.append(meteor)

        print("BLEU: ", bleu)
        print("Average BLEU so far: ", sum(bleus) / len(bleus))
        print("METEOR: ", meteor)
        print("Average METEOR so far: ", sum(meteors) / len(meteors))
    print("Average BLEU score:")
    print(sum(bleus) / len(bleus))
    print("Average max possible BLEU score:")
    print(sum(max_bleus) / len(max_bleus))
    print("Average METEOR: ", sum(meteors) / len(meteors))


test_model_chars("delex_test", delex_slots=("name", "priceRange", "near", "area"))